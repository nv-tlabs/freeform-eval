# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from NVIDIA Kaolin
# (https://github.com/NVIDIAGameWorks/kaolin) and have been modified.
# FreeformSimulatedObject and FreeformScene subclass and adapt Kaolin's
# SimulatedObject and SimplicitsScene from physics/simplicits/simulation.py;
# _freeform_newtons_method is adapted from newtons_method in
# physics/common/optimization.py. Changes include:
# - Scalar rather than per-column skinning weight normalization
# - QR rank truncation to numerical rank, with _dof_bs=1 instead of 4
# - Separate boundary conditions with their own LBS matrix
# - BSR-to-torch conversion fix for (1,1) block shapes

"""
Kaolin Simplicits extensions. Overloads marked with ############### in the code.

1. Scalar weight normalization: Skinning weights are normalized with a single scalar
   for consistent conditioning of the Newton system. Kaolin uses per-column L2
   normalization which can produce different QR bases and numerically different
   simulation results. Set normalize_weights_scalar=True (default) to reproduce
   the exact results in the paper.

2. QR rank truncation: The QR decomposition is truncated to numerical rank, removing
   near-zero modes. Released kaolin keeps all columns. Set truncate_qr_rank=True to
   reproduce the exact results in the paper. This also requires _dof_bs=1 (not 4)
   since the truncated rank may not be a multiple of 4, and uses _warp_csr instead
   of bsr_copy for building sparse matrices.

3. Separate boundary conditions: Boundary points with their own LBS matrix,
   independent from quadrature points. Adds penalty energy/gradient/hessian
   contributions via set_object_separate_boundary_condition.

4 (minor). BSR conversion fix: Kaolin's _wp_bsr_to_torch_bsr fails for (1,1) block shape
   (values not flattened). We reimplement the conversion (wp_bsr_to_torch) and use
   our own Newton solver (_freeform_newtons_method) that calls it.
"""

import numpy as np
import torch
import warp as wp
import warp.sparse as wps
from scipy.linalg import qr

from kaolin.physics.simplicits import SimplicitsScene, SimplicitsObject
from kaolin.physics.simplicits.simulation import SimulatedObject
from kaolin.physics.simplicits.precomputed import sparse_lbs_matrix
from kaolin.physics.simplicits.training import PhysicsPoints
from kaolin.physics.common import Boundary
from kaolin.physics.utils import warp_utilities, torch_utilities
from kaolin.physics.materials.material_utils import get_defo_grad, to_lame
from kaolin.physics.materials import NeohookeanElasticMaterial


def wp_bsr_to_torch(mat):
    """Convert warp BSR/CSR matrix to torch sparse, handling (1,1) blocks correctly.
    Fix for kaolin's _wp_bsr_to_torch_bsr which fails for (1,1) blocks (values not flattened).
    """
    mat.nnz_sync()
    if mat.block_shape == (1, 1):
        return torch.sparse_csr_tensor(
            crow_indices=wp.to_torch(mat.offsets[: mat.nrow + 1]),
            col_indices=wp.to_torch(mat.columns[: mat.nnz]),
            values=wp.to_torch(mat.values[: mat.nnz]).flatten(),
            size=mat.shape)
    return torch.sparse_bsr_tensor(
        crow_indices=wp.to_torch(mat.offsets[: mat.nrow + 1]),
        col_indices=wp.to_torch(mat.columns[: mat.nnz]),
        values=wp.to_torch(mat.values[: mat.nnz]),
        size=mat.shape)


from kaolin.physics.common.optimization import _red_to_full, _full_to_red, _line_search, _apply_bounds
import logging as _logging

_newton_logger = _logging.getLogger(__name__)

@torch.no_grad()
def _freeform_newtons_method(x, energy_fcn, gradient_fcn, hessian_fcn,
                             bounds_fcn, Pt, P, nm_max_iters, conv_tol,
                             direct_solve=True):
    t_x_kin = wp.to_torch(x) - wp.to_torch(_red_to_full(P, _full_to_red(Pt, x)))

    for k in range(nm_max_iters):
        E = energy_fcn(x)
        G = gradient_fcn(x).flatten()
        H = hessian_fcn(x)

        if P is not None:
            red_H = Pt @ H @ P
            red_g = Pt @ G
        else:
            red_H = H
            red_g = G

        t_red_x = wp.to_torch(_full_to_red(Pt, x))
        t_red_g = wp.to_torch(red_g)

        if direct_solve:
            A = wp_bsr_to_torch(red_H).to_dense()
            b = wp.to_torch(red_g)
            t_red_dx = -torch.linalg.solve(A, b)
        else:
            from warp.optim.linear import cg, preconditioner
            precond = preconditioner(red_H, "diag")
            dx = wp.zeros_like(red_g)
            cg(A=red_H, b=red_g, x=dx, tol=1e-4, maxiter=100, M=precond)
            t_red_dx = -wp.to_torch(dx)

        if torch.abs(t_red_dx.t() @ t_red_g) < conv_tol:
            _newton_logger.debug(f"Newton: Converged in {k} iterations")
            break

        full_dx = _red_to_full(P, wp.from_torch(t_red_dx))
        wp_bounds = bounds_fcn(full_dx, x)
        if wp_bounds is None:
            t_bounds = torch.ones_like(t_red_x)
        else:
            t_bounds = wp.to_torch(_full_to_red(Pt, wp_bounds))

        bounded_update = _line_search(
            func=energy_fcn, x=t_red_x, wp_P=P,
            direction=t_red_dx, gradient=t_red_g,
            initial_step_size=1.0, bounds=t_bounds,
            bounds_qr_tfm=None, bounds_qr_tfm_inv=None)

        t_red_x = t_red_x + bounded_update
        x = wp.from_torch(wp.to_torch(_red_to_full(P, wp.from_torch(t_red_x))) + t_x_kin)
    else:
        _newton_logger.debug(f"Newton: Not converged in {nm_max_iters} iterations")

    return x


# ---------------------------------------------------------------------------
# FreeformSimulatedObject
# ---------------------------------------------------------------------------

class FreeformSimulatedObject(SimulatedObject):
    """SimulatedObject with optional QR rank truncation and scalar weight normalization."""

    QR_RANK_TOL = 1e-5

    def __init__(self, *args, truncate_qr_rank=False, normalize_weights_scalar=False, **kwargs):
        self.truncate_qr_rank = truncate_qr_rank
        self._num_dof = None
        self._normalize_weights_scalar = normalize_weights_scalar
        self._scalar_w_norm = None
        self._scalar_one_norm = None

        ############### Scalar weight normalization (old SkinningWeightsFcn style) ###############
        if normalize_weights_scalar:
            sw = kwargs.get('skinning_weights')
            if sw is not None:
                w_raw = sw[:, :-1]
                ones_col = sw[:, -1:]
                self._scalar_w_norm = torch.sqrt(torch.diag(w_raw.T @ w_raw).mean()).clamp(min=1e-10)
                self._scalar_one_norm = torch.sqrt((ones_col ** 2).sum()).clamp(min=1e-10)
                kwargs['skinning_weights'] = torch.cat([
                    w_raw / self._scalar_w_norm, ones_col / self._scalar_one_norm], dim=1)
                dwdx = kwargs.get('dwdx')
                if dwdx is not None:
                    dwdx = dwdx.clone()
                    dwdx[:, :-1, :] /= self._scalar_w_norm
                    dwdx[:, -1:, :] /= self._scalar_one_norm
                    kwargs['dwdx'] = dwdx
            kwargs['normalize_weights_by_samples'] = False
        ###############################################################################

        super().__init__(*args, **kwargs)

    @property
    def num_dof(self):
        if self._num_dof is not None:
            return self._num_dof
        return self.num_handles * 12

    @classmethod
    def from_skinned_physics_points(cls, phys_pts, init_transform, is_kinematic=False,
                                    normalize_weights_by_samples=False, apply_qr=False,
                                    truncate_qr_rank=False, normalize_weights_scalar=False):
        return cls(pts=phys_pts.pts, yms=phys_pts.yms, prs=phys_pts.prs,
                   rhos=phys_pts.rhos, appx_vol=phys_pts.appx_vol,
                   skinning_weights=phys_pts.skinning_weights, dwdx=phys_pts.dwdx,
                   renderable=phys_pts.renderable,
                   init_transform=init_transform, is_kinematic=is_kinematic,
                   normalize_weights_by_samples=normalize_weights_by_samples,
                   apply_qr=apply_qr, truncate_qr_rank=truncate_qr_rank,
                   normalize_weights_scalar=normalize_weights_scalar)

    def _apply_qr_decomposition(self):
        if not self.truncate_qr_rank:
            return super()._apply_qr_decomposition()

        B_dense = self.B_dense
        np_B = B_dense.detach().cpu().numpy()
        _, np_R, np_P = qr(np_B, mode='economic', pivoting=True)

        pmat = torch.eye(np_B.shape[1], device=self.device, dtype=self.dtype)[:, np_P]
        R = torch.from_numpy(np_R).to(device=self.device, dtype=self.dtype)

        ############### QR rank truncation to numerical rank ###############
        rank = int(np.sum(np.abs(np.diag(np_R)) > self.QR_RANK_TOL))
        self._num_dof = rank
        self.qr_tfm = pmat @ torch.linalg.solve_triangular(
            R, torch.eye(R.shape[0], device=R.device, dtype=R.dtype), upper=True)[:, :rank]
        self.qr_tfm_inv = R[:rank, :] @ pmat.T
        ###################################################################

        ############### Use _warp_csr (not bsr_copy) — rank may not be multiple of 4 ###############
        Q_dense = B_dense @ self.qr_tfm
        self.B = warp_utilities._warp_csr_from_torch_dense(Q_dense)
        self.B.nnz_sync()
        self._B_dense = Q_dense

        if not self.is_kinematic:
            dFdz_dense = self.dFdz_dense
            self._dFdz_dense = dFdz_dense @ self.qr_tfm
            self.dFdz = warp_utilities._warp_csr_from_torch_dense(self._dFdz_dense)
            self.dFdz.nnz_sync()
        ###########################################################################################

    def reset_sim_state(self):
        if self._num_dof is None:
            return super().reset_sim_state()
        orig_dof = self.skinning_weights.shape[1] * 12
        z_pre_qr = torch.zeros(orig_dof, dtype=self.dtype, device=self.device)
        if self.init_transform is not None:
            ############### Scalar norm: use _scalar_one_norm for init scale ###############
            if self._normalize_weights_scalar and self._scalar_one_norm is not None:
                scale = self._scalar_one_norm.detach()
            ###############################################################################
            elif self.normalize_weights_by_samples:
                scale = self.handle_norms[-1].detach()
            else:
                scale = 1.0
            z_pre_qr[-12:] = self.init_transform.flatten() * scale
        ############### qr_tfm_inv (supports truncated rank) ###############
        self.z = self.qr_tfm_inv @ z_pre_qr if self.apply_qr else z_pre_qr
        ###################################################################
        self.z_prev = self.z.clone().detach()
        self.z_dot = torch.zeros_like(self.z, device=self.device)


# ---------------------------------------------------------------------------
# FreeformScene
# ---------------------------------------------------------------------------

class FreeformScene(SimplicitsScene):
    """SimplicitsScene with separate boundary conditions and QR rank truncation support."""

    def _compute_sim_constants(self):
        _stacked_pts = []
        _stacked_rhos = []
        _stacked_vols = []
        _stacked_masses = []
        _stacked_yms = []
        _stacked_prs = []
        _stacked_sparse_B = []
        _stacked_sparse_dFdz = []
        _stacked_skinning_weights = []

        _object_to_z_map = {}
        _object_to_qp_map = {}
        _qp_to_object_map = []
        _z_to_object_map = []
        _kin_obj_list = []
        _kin_obj_to_z_map = {}
        _kin_obj_to_qp_map = {}

        z_index = 0
        x_index = 0
        for object_id, obj in self.sim_obj_dict.items():
            ############### Use num_dof instead of num_handles*12 ###############
            num_dof = obj.num_dof if hasattr(obj, 'num_dof') else obj.num_handles * 12
            ###############################################################

            _object_to_z_map[object_id] = wp.array(
                np.arange(z_index, z_index + num_dof), dtype=wp.int32)
            _object_to_qp_map[object_id] = wp.array(
                np.arange(x_index, x_index + obj.num_qp), dtype=wp.int32)
            _qp_to_object_map.append(torch.full(
                (obj.num_qp,), object_id, dtype=torch.int32, device=self.device))
            _z_to_object_map.append(torch.full(
                (num_dof,), object_id, dtype=torch.int32))

            if obj.is_kinematic:
                _kin_obj_list.append(object_id)
                _kin_obj_to_z_map[object_id] = wp.array(
                    np.arange(z_index, z_index + num_dof), dtype=wp.int32)
                _kin_obj_to_qp_map[object_id] = wp.array(
                    np.arange(x_index, x_index + obj.num_qp), dtype=wp.int32)

            z_index += num_dof
            x_index += obj.num_qp

            _stacked_pts.append(obj.pts)
            _stacked_rhos.append(obj.rhos)
            _stacked_vols.append(obj.sample_vols)
            _stacked_masses.append(obj.sample_masses)
            _stacked_yms.append(obj.yms)
            _stacked_prs.append(obj.prs)
            _stacked_sparse_B.append(obj.B)
            _stacked_sparse_dFdz.append(obj.dFdz)
            _stacked_skinning_weights.append(obj.skinning_weights)

        self.object_to_z_map = _object_to_z_map
        self.object_to_qp_map = _object_to_qp_map
        self.qp_to_object_map = wp.from_torch(torch.cat(_qp_to_object_map))
        self.z_to_object_map = wp.from_torch(torch.cat(_z_to_object_map))
        self.kin_obj_list = _kin_obj_list
        self.kin_obj_to_z_map = _kin_obj_to_z_map
        self.kin_obj_to_qp_map = _kin_obj_to_qp_map

        t_qp_is_kinematic = torch.zeros(x_index, dtype=torch.int32, device=self.device)
        for kin_id in _kin_obj_list:
            t_qp_is_kinematic[wp.to_torch(self.object_to_qp_map[kin_id])] = 1
        self.qp_is_kinematic = wp.from_torch(t_qp_is_kinematic)

        self.sim_pts = wp.from_torch(torch.cat(_stacked_pts, dim=0).contiguous(), dtype=wp.vec3)
        self.sim_pts_flat = wp.from_torch(torch.cat(_stacked_pts, dim=0).flatten().contiguous())
        self.sim_skinning_weights = wp.from_torch(torch.cat(_stacked_skinning_weights, dim=0).contiguous())
        self.sim_rhos = wp.from_torch(torch.cat(_stacked_rhos, dim=0).contiguous())
        self.sim_vols = wp.from_torch(torch.cat(_stacked_vols, dim=0).contiguous())

        ############### _dof_bs=1 for truncated QR (arbitrary rank) ###############
        has_truncation = any(hasattr(obj, 'truncate_qr_rank') and obj.truncate_qr_rank
                            for obj in self.sim_obj_dict.values())
        self._dof_bs = 1 if has_truncation else 4
        ###################################################################

        ############### Skip bsr_copy when _dof_bs=1 (truncated QR) ###############
        kin_dofs = [wp.to_torch(self.kin_obj_to_z_map[kid]) for kid in _kin_obj_list]
        if len(kin_dofs) > 0:
            temp_Pt = warp_utilities._warp_csr_from_torch_dense(
                torch_utilities.create_projection_matrix(z_index, torch.cat(kin_dofs, dim=0)))
            if self._dof_bs > 1:
                self.sim_Pt = wps.bsr_copy(temp_Pt, block_shape=(1, self._dof_bs))
            else:
                self.sim_Pt = temp_Pt
            self.sim_P = self.sim_Pt.transpose()
        else:
            self.sim_Pt = None
            self.sim_P = None

        wp_masses = wp.from_torch(torch.cat(_stacked_masses, dim=0).contiguous())
        np_masses = np.repeat(wp_masses.numpy(), 3)
        self.sim_M = wps.bsr_diag(wp.array(np_masses, dtype=wp.float32))

        temp_B = warp_utilities._block_diagonalize(_stacked_sparse_B)
        if self._dof_bs > 1:
            self.sim_B = wps.bsr_copy(temp_B, block_shape=(1, self._dof_bs))
        else:
            self.sim_B = temp_B
        self.sim_BMB = wps.bsr_transposed(self.sim_B) @ self.sim_M @ self.sim_B

        temp_dFdz = warp_utilities._block_diagonalize(_stacked_sparse_dFdz)
        if self._dof_bs > 1:
            self.sim_dFdz = wps.bsr_copy(temp_dFdz, block_shape=(9, self._dof_bs))
        else:
            self.sim_dFdz = temp_dFdz
        ###########################################################################

        self.sim_M.nnz_sync()
        self.sim_B.nnz_sync()
        self.sim_dFdz.nnz_sync()
        self.sim_BMB.nnz_sync()

        ############### Use qr_tfm/qr_tfm_inv (supports truncated rank) ###############
        if any(obj.apply_qr for obj in self.sim_obj_dict.values()):
            _rinv_blocks = []
            _rinv_red_blocks = []
            _rinv_red_inv_blocks = []
            for obj in self.sim_obj_dict.values():
                num_dof = obj.num_dof if hasattr(obj, 'num_dof') else obj.num_handles * 12
                if obj.apply_qr:
                    _rinv_blocks.append(obj.qr_tfm)
                    if not obj.is_kinematic:
                        _rinv_red_blocks.append(obj.qr_tfm)
                        _rinv_red_inv_blocks.append(obj.qr_tfm_inv)
                else:
                    eye = torch.eye(num_dof, device=self.device, dtype=self.dtype)
                    _rinv_blocks.append(eye)
                    if not obj.is_kinematic:
                        _rinv_red_blocks.append(eye)
                        _rinv_red_inv_blocks.append(eye)
            self.sim_qr_tfm = torch.block_diag(*_rinv_blocks).contiguous()
            self.sim_qr_tfm_red = torch.block_diag(*_rinv_red_blocks).contiguous() if _rinv_red_blocks else None
            self.sim_qr_tfm_inv_red = torch.block_diag(*_rinv_red_inv_blocks).contiguous() if _rinv_red_inv_blocks else None
        else:
            self.sim_qr_tfm = None
            self.sim_qr_tfm_red = None
            self.sim_qr_tfm_inv_red = None
        ###############################################################################

        # Material
        mus, lams = to_lame(
            torch.cat(_stacked_yms, dim=0).contiguous(),
            torch.cat(_stacked_prs, dim=0).contiguous())
        self.sim_mus = wp.from_torch(mus)
        self.sim_lams = wp.from_torch(lams)

        elastic_struct = NeohookeanElasticMaterial(
            mu=self.sim_mus, lam=self.sim_lams,
            integration_pt_volume=self.sim_vols, reparameterize_lame=True)
        self.force_dict["defo_grad_wise"]["material"] = {}
        self.force_dict["defo_grad_wise"]["material"]["object"] = elastic_struct
        self.force_dict["defo_grad_wise"]["material"]["coeff"] = 1.0

    def add_object(self, sim_object, num_qp=None, init_transform=None, is_kinematic=False,
                   renderable_pts=None, normalize_weights_by_samples=False, apply_qr=False,
                   truncate_qr_rank=False, normalize_weights_scalar=False):
        if torch.is_tensor(init_transform):
            relative_transform = torch_utilities.standard_transform_to_relative(init_transform)
        else:
            relative_transform = torch.zeros(3, 4, device=self.device, dtype=self.dtype)

        if isinstance(sim_object, SimplicitsObject):
            assert num_qp is not None
            baked = sim_object.bake(num_qps=num_qp, renderable_pts=renderable_pts)
            simulated_object = FreeformSimulatedObject.from_skinned_physics_points(
                baked, init_transform=relative_transform, is_kinematic=is_kinematic,
                normalize_weights_by_samples=normalize_weights_by_samples,
                apply_qr=apply_qr, truncate_qr_rank=truncate_qr_rank,
                normalize_weights_scalar=normalize_weights_scalar)
        else:
            sampled = sim_object.subsample(num_pts=num_qp) if num_qp is not None else sim_object
            simulated_object = FreeformSimulatedObject.from_skinned_physics_points(
                sampled, init_transform=relative_transform, is_kinematic=is_kinematic,
                normalize_weights_by_samples=normalize_weights_by_samples,
                apply_qr=apply_qr, truncate_qr_rank=truncate_qr_rank,
                normalize_weights_scalar=normalize_weights_scalar)

        return self._add_object(simulated_object)

    def get_object_transforms(self, object_id):
        tfms = self._get_object_transforms_internal(object_id)
        obj = self.sim_obj_dict[object_id]
        ############### Un-normalize transforms for scalar normalization ###############
        if obj._normalize_weights_scalar:
            norms = torch.ones(tfms.shape[0], device=tfms.device, dtype=tfms.dtype)
            norms[:-1] = obj._scalar_w_norm
            norms[-1] = obj._scalar_one_norm
            tfms[:, :3, :] = tfms[:, :3, :] / norms.view(-1, 1, 1)
        ###############################################################################
        elif obj.normalize_weights_by_samples:
            tfms[:, :3, :] = tfms[:, :3, :] / obj.handle_norms.view(-1, 1, 1)
        return tfms

    def get_object_deformed_pts(self, obj_idx, points='simulated'):
        obj = self.get_object(obj_idx)
        if not (hasattr(obj, 'truncate_qr_rank') and obj.truncate_qr_rank):
            return super().get_object_deformed_pts(obj_idx, points)

        from kaolin.physics.simplicits.skinning import standard_lbs
        wp_z = wp.clone(self.sim_z[self.object_to_z_map[obj_idx]])
        z_truncated = wp.to_torch(wp_z, requires_grad=False)
        z_original = obj.qr_tfm @ z_truncated  ############### qr_tfm (truncated rank) ###############
        tfms = z_original.reshape(-1, 3, 4)

        if points == 'simulated':
            pts = obj.pts
            sw = obj.skinning_weights
        elif points == 'rendered':
            pts = obj.renderable.pts
            sw = obj.renderable.skinning_weights
            ############### Un-normalize transforms for raw renderable weights ###############
            if obj._normalize_weights_scalar:
                norms = torch.ones(tfms.shape[0], device=tfms.device, dtype=tfms.dtype)
                norms[:-1] = obj._scalar_w_norm
                norms[-1] = obj._scalar_one_norm
                tfms[:, :3, :] = tfms[:, :3, :] / norms.view(-1, 1, 1)
            ###############################################################################
            elif obj.normalize_weights_by_samples:
                tfms[:, :3, :] = tfms[:, :3, :] / obj.handle_norms.view(-1, 1, 1)
        else:
            raise ValueError('Only "rendered" or "simulated"')

        padding = torch.zeros(tfms.shape[0], 1, 4, device=tfms.device, dtype=tfms.dtype)
        padding[:, 0, 3] = 1.0
        tfms = torch.cat([tfms, padding], dim=1)

        return standard_lbs(pts, tfms[:, :3, :].unsqueeze(0), sw).squeeze()

    def run_sim_step(self):
        has_truncation = any(
            hasattr(obj, 'truncate_qr_rank') and obj.truncate_qr_rank
            for obj in self.sim_obj_dict.values())

        if not has_truncation:
            return super().run_sim_step()

        if not self._ready_for_forces:
            raise RuntimeError("Forces need to be set")
        self._detect_collision(self.sim_z)
        self.sim_z_prev = wp.clone(self.sim_z)

        from functools import partial
        E_fn = partial(self._newton_E, wp_B=self.sim_B, dt=self.timestep,
                        wp_z_prev=self.sim_z_prev, wp_z_dot=self.sim_z_dot)
        G_fn = partial(self._newton_G, wp_B=self.sim_B, wp_BMB=self.sim_BMB,
                        dt=self.timestep, wp_z_prev=self.sim_z_prev, wp_z_dot=self.sim_z_dot)
        H_fn = partial(self._newton_H, wp_B=self.sim_B, wp_BMB=self.sim_BMB, dt=self.timestep)

        ############### Use our Newton solver with correct BSR conversion ###############
        self.sim_z = _freeform_newtons_method(
            self.sim_z, E_fn, G_fn, H_fn,
            bounds_fcn=self._compute_collision_bounds,
            Pt=self.sim_Pt, P=self.sim_P,
            nm_max_iters=self.max_newton_steps,
            conv_tol=self.conv_tol,
            direct_solve=self.direct_solve)
        ###############################################################################

        self.sim_z_dot = wp.from_torch(
            (wp.to_torch(self.sim_z) - wp.to_torch(self.sim_z_prev)) / self.timestep)
        self.current_sim_step += 1

    def set_object_separate_boundary_condition(self, obj_idx, bdry_pos, skinning_mod,
                                               bdry_penalty=10000.0, pinned_x=None):
        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()

        boundary_struct = Boundary(
            integration_pt_volume=wp.ones((bdry_pos.shape[0],), dtype=wp.float32))
        bdry_indx = wp.array(list(range(bdry_pos.shape[0])), dtype=wp.int32)
        if pinned_x is None:
            pinned_x = bdry_pos.clone()
        boundary_struct.set_pinned(indices=bdry_indx, pinned_x=wp.from_torch(pinned_x, dtype=wp.vec3))

        if "separate_boundary" not in self.force_dict:
            self.force_dict["separate_boundary"] = {}
        self.force_dict["separate_boundary"][obj_idx] = {}
        self.force_dict["separate_boundary"][obj_idx]["object"] = boundary_struct
        self.force_dict["separate_boundary"][obj_idx]["coeff"] = bdry_penalty

        obj = self.get_object(obj_idx)
        bdry_skinning_weights = skinning_mod.compute_skinning_weights(bdry_pos)
        ############### Normalize boundary weights to match quadrature normalization ###############
        if obj._normalize_weights_scalar:
            w_raw = bdry_skinning_weights[:, :-1]
            ones_col = bdry_skinning_weights[:, -1:]
            bdry_skinning_weights = torch.cat([
                w_raw / obj._scalar_w_norm,
                ones_col / obj._scalar_one_norm
            ], dim=1)
        elif obj.normalize_weights_by_samples and obj.handle_norms is not None:
            bdry_skinning_weights = bdry_skinning_weights / obj.handle_norms.unsqueeze(0)
        ###########################################################################################
        bdry_pts_B = sparse_lbs_matrix(
            wp.from_torch(bdry_skinning_weights), wp.from_torch(bdry_pos, dtype=wp.vec3))
        bdry_pts_B_dense = warp_utilities._bsr_to_torch(bdry_pts_B).to_dense()
        if obj.apply_qr:
            bdry_pts_B_dense = bdry_pts_B_dense @ obj.qr_tfm  ############### qr_tfm (truncated rank) ###############

        self.force_dict["separate_boundary"][obj_idx]["bdry_pts_B_dense_torch"] = bdry_pts_B_dense
        self.force_dict["separate_boundary"][obj_idx]["bdry_pos"] = wp.from_torch(bdry_pos, dtype=wp.vec3)
        return pinned_x

    def _get_separate_boundary_dx(self, obj_idx, z):
        d = self.force_dict["separate_boundary"][obj_idx]
        obj_z = wp.to_torch(z)[wp.to_torch(self.object_to_z_map[obj_idx])]
        return wp.from_torch((d["bdry_pts_B_dense_torch"] @ obj_z).view(-1, 3), dtype=wp.vec3)

    def _assemble_energies(self, z, delta_dz):
        pe, ke = super()._assemble_energies(z, delta_dz)
        if "separate_boundary" in self.force_dict:
            for oi, od in self.force_dict["separate_boundary"].items():
                dx = self._get_separate_boundary_dx(oi, self._eval_z)
                e = wp.zeros(1, dtype=wp.float32)
                od["object"].energy(dx, od["bdry_pos"], od["coeff"], e)
                pe += e.numpy()[0]
        return pe, ke

    def _assemble_gradients(self, z):
        g = super()._assemble_gradients(z)
        if "separate_boundary" in self.force_dict:
            gt = wp.to_torch(g)
            for oi, od in self.force_dict["separate_boundary"].items():
                dx = self._get_separate_boundary_dx(oi, self._eval_z)
                zi = wp.to_torch(self.object_to_z_map[oi])
                bg = od["object"].gradient(dx, od["bdry_pos"], od["coeff"], None)
                gt[zi] += od["bdry_pts_B_dense_torch"].T @ wp.to_torch(bg).view(-1)
        return g

    def _assemble_hessians(self, z):
        num_pts = int(self.sim_B.shape[0] / 3)
        pt_names = list(self.force_dict["pt_wise"].keys())
        dg_names = list(self.force_dict["defo_grad_wise"].keys())

        F = get_defo_grad(z, self.sim_dFdz)
        dx = wp.array(self.sim_B @ wp.array(z, dtype=wp.vec3), dtype=wp.vec3)
        x0 = self.sim_pts

        d2x = torch.zeros(num_pts, 3, 3, device=self.device, dtype=self.dtype)
        for e in pt_names:
            fo = self.force_dict["pt_wise"][e]
            d2x += wp.to_torch(fo["object"].hessian(dx, x0, fo["coeff"]), requires_grad=False)

        d2F = torch.zeros(num_pts, 9, 9, device=self.device, dtype=self.dtype)
        for e in dg_names:
            fo = self.force_dict["defo_grad_wise"][e]
            d2F += wp.to_torch(fo["object"].hessian(F, fo["coeff"]), requires_grad=False)

        hess_list = []
        for oid, obj in self.sim_obj_dict.items():
            qi = wp.to_torch(self.object_to_qp_map[oid])
            H = (torch_utilities.hess_reduction(obj.B_dense, d2x[qi]) +
                 torch_utilities.hess_reduction(obj.dFdz_dense, d2F[qi]))
            ############### Add separate boundary hessian contribution ###############
            if "separate_boundary" in self.force_dict and oid in self.force_dict["separate_boundary"]:
                od = self.force_dict["separate_boundary"][oid]
                bdx = self._get_separate_boundary_dx(oid, self._eval_z)
                bh = wp.to_torch(od["object"].hessian(bdx, od["bdry_pos"], od["coeff"]))
                H += torch_utilities.hess_reduction(od["bdry_pts_B_dense_torch"], bh)
            ###################################################################
            hess_list.append((oid, oid, H))

        if "collision" in self.force_dict and self.force_dict["collision"]["object"].num_contacts > 0:
            cs = self.force_dict["collision"]["object"]
            cc = self.force_dict["collision"]["coeff"]
            ch = wp.to_torch(cs.hessian(dx, x0, cc), requires_grad=False)
            od = self.object_to_z_map
            for i, j in cs.object_pairs:
                Ji = cs.collision_J_dense[:, wp.to_torch(od[i])]
                Jj = cs.collision_J_dense[:, wp.to_torch(od[j])]
                hess_list.append((i, j, torch_utilities.hess_reduction(Ji, ch, Jj)))

        return warp_utilities._assemble_global_hessian(
            hess_list, self.object_to_z_map, z, block_size=self._dof_bs)


__all__ = ['FreeformSimulatedObject', 'FreeformScene', 'SimplicitsObject', 'PhysicsPoints']
