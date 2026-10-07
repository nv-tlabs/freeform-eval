# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Run Simplicits simulation using saved skinning weights.

Usage:
    uv run python sim/run_sim.py --config config/thingi10k/80597/sim_rkpm_fix_right.yaml
"""

import argparse
import os
import sys
import time
import logging

import numpy as np
from scipy.spatial import cKDTree
import torch
import warp as wp

import kaolin
from kaolin.physics.simplicits import SimplicitsObject
from kaolin.physics.simplicits.training import PhysicsPoints
from omegaconf import OmegaConf
import trimesh as _trimesh

# Add project root to path for sim.scene import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.simplicits_ext import FreeformScene
from sim.utils import sample_interior_uniform_grid

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Simplicits simulation')
    parser.add_argument('--config', type=str, required=True, help='Path to sim config yaml')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['INFO', 'DEBUG'])
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config values, e.g. model=path/to/model.pth')
    args = parser.parse_args()

    if args.log_level == 'DEBUG':
        logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = 'cuda'

    # Number of simulation steps
    num_steps = int(cfg.get("num_steps", 100))

    # Load boundary config (optional)
    bdry_cfg = None
    if cfg.get("boundary_config", None) is not None:
        bdry_cfg = OmegaConf.load(os.path.join(project_root, cfg.boundary_config))
        bdry_num_frames = bdry_cfg.get("n_frames", num_steps)
        assert num_steps == bdry_num_frames, \
            f"num_steps ({num_steps}) != boundary n_frames ({bdry_num_frames})"

    # Load saved model
    model_path = os.path.join(project_root, cfg.model)
    saved_model = torch.load(model_path, weights_only=False)
    so_model = saved_model['model']
    so_model.to(device=device)

    # Load mesh
    original_mesh_file = saved_model["input_mesh_file"]
    tri_mesh = _trimesh.load(os.path.join(project_root, original_mesh_file), process=False)
    mesh = kaolin.rep.SurfaceMesh(
        vertices=torch.tensor(tri_mesh.vertices, device=device, dtype=torch.float32),
        faces=torch.tensor(tri_mesh.faces, device=device, dtype=torch.long),
    )
    orig_vertices = mesh.vertices.clone()

    # Sample quadrature points (uniform grid)
    NUM_SAMPLES = int(cfg.get("num_samples", 125000))
    so_pts, _ = sample_interior_uniform_grid(mesh, NUM_SAMPLES, device)

    # Material properties
    materials_path = saved_model.get('materials', None)
    if materials_path is not None:
        # Spatially varying materials via nearest-neighbor from VoMP npz
        mat_data = np.load(os.path.join(project_root, materials_path))
        voxel_data = mat_data["voxel_data"]
        mat_pts = np.stack([voxel_data["x"], voxel_data["y"], voxel_data["z"]], axis=1)
        tree = cKDTree(mat_pts)
        _, nn_idx = tree.query(so_pts.cpu().numpy(), k=1)
        so_yms = torch.tensor(voxel_data["youngs_modulus"][nn_idx], device=device, dtype=torch.float32)
        so_prs = torch.tensor(voxel_data["poissons_ratio"][nn_idx], device=device, dtype=torch.float32)
        so_rhos = torch.tensor(voxel_data["density"][nn_idx], device=device, dtype=torch.float32)
        print(f"Spatially varying materials: YM [{so_yms.min():.2e}, {so_yms.max():.2e}]")
    else:
        # Uniform materials
        so_yms = saved_model['so_yms'][0].squeeze() * torch.ones(so_pts.shape[0], device=device)
        so_prs = saved_model['so_prs'][0].squeeze() * torch.ones(so_pts.shape[0], device=device)
        so_rhos = saved_model['so_rhos'][0].squeeze() * torch.ones(so_pts.shape[0], device=device)
    so_appx_vol = saved_model['so_appx_vol']
    print(f"Quadrature points: {so_pts.shape[0]}")

    # Create SimplicitsObject and scene
    physics_pts = PhysicsPoints(pts=so_pts, yms=so_yms, prs=so_prs, rhos=so_rhos, appx_vol=so_appx_vol)
    sim_obj = SimplicitsObject.create_from_function(physics_points=physics_pts, fcn=so_model)

    dt = float(cfg.get("dt", 0.05))
    scene = FreeformScene(
        device=device,
        timestep=dt,
        max_newton_steps=int(cfg.get("max_newton_steps", 20)),
    )
    scene.conv_tol = float(cfg.get("conv_tol", 1e-8))
    # Default 0.01: the unclamped Neo-Hookean Hessian can come out near-singular
    # on soft examples (GPU-atomics nondeterminism decides), making the Newton
    # solve emit an enormous step that explodes the state in ~25% of runs
    # (observed on Thingi10K 96123 pull_boundary, ym=1e4). A small diagonal
    # regularizer suppresses this; 0.01 leaves results unchanged (median
    # per-example MSE shift 0.025% over all 120 Thingi10K configs).
    scene.newton_hessian_regularizer = float(cfg.get("newton_hessian_regularizer", 0.01))
    scene.direct_solve = cfg.get("direct_solve", True)

    apply_qr = cfg.get("apply_qr", True)
    truncate_qr = cfg.get("truncate_qr_rank", False)
    normalize_scalar = cfg.get("normalize_weights_scalar", True)
    scene.add_object(sim_obj, num_qp=NUM_SAMPLES, apply_qr=apply_qr,
                     renderable_pts=orig_vertices,
                     truncate_qr_rank=truncate_qr,
                     normalize_weights_scalar=normalize_scalar)
    scene.set_scene_gravity(torch.tensor([0, 9.8, 0], device=device))

    # Separate boundary condition (optional)
    bdry_pos_seq = None
    if bdry_cfg is not None:
        bdry_file = os.path.join(project_root, bdry_cfg["boundary_file"])
        bdry_data = np.load(bdry_file)
        assert bdry_data["num_frames"] == num_steps
        bdry_rest_pos = torch.tensor(bdry_data["boundary_rest_pos"], device=device, dtype=torch.float32)
        bdry_pos_seq = torch.tensor(bdry_data["boundary_pos_seq"], device=device, dtype=torch.float32)
        pinned_x = bdry_pos_seq[0].clone()
        bdry_weight = float(cfg.get("separate_bdry_weight", 1e4))
        scene.set_object_separate_boundary_condition(0, bdry_rest_pos, skinning_mod=so_model,
                                                      bdry_penalty=bdry_weight, pinned_x=pinned_x)

    # Run simulation
    states = [wp.to_torch(wp.clone(scene.sim_z)).squeeze()]
    xt_pts = [scene.get_object_deformed_pts(0, points='simulated').squeeze()]

    print(f"\nRunning {num_steps} steps (dt={dt})...")
    start_time = time.time()
    for i in range(num_steps):
        if bdry_pos_seq is not None:
            wp.copy(
                scene.force_dict["separate_boundary"][0]["object"].pinned_vertices,
                wp.from_torch(bdry_pos_seq[i + 1], dtype=wp.vec3)
            )
        print(f"--- Step {i} ---")
        scene.run_sim_step()
        states.append(wp.to_torch(wp.clone(scene.sim_z)).squeeze())
        xt_pts.append(scene.get_object_deformed_pts(0, points='simulated').squeeze())
    elapsed = time.time() - start_time
    print(f"Simulation time: {elapsed:.2f}s")

    # Deform original mesh vertices
    print("Computing deformed mesh vertices...")
    xt_orig_vertices = []
    for z_state in states:
        with torch.no_grad():
            scene.sim_z.assign(wp.from_torch(z_state))
            deformed = scene.get_object_deformed_pts(0, points='rendered')
            xt_orig_vertices.append(deformed.cpu())

    # Save
    output_dir = os.path.join(project_root, cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(cfg.model))[0]
    if cfg.get("boundary_config", None) is not None:
        bdry_config_name = os.path.splitext(os.path.basename(cfg.boundary_config))[0]
        output_path = os.path.join(output_dir, f"sim_result_{model_name}_{bdry_config_name}.pth")
    else:
        output_path = os.path.join(output_dir, f"sim_result_{model_name}.pth")

    save_dict = {
        # Per-frame DOF state vectors z, shape (num_frames+1, num_handles*12)
        "states": [x.cpu() for x in states],
        # Per-frame deformed quadrature points, shape (num_frames+1, num_qp, 3)
        "xt_pts": [x.cpu() for x in xt_pts],
        # Per-frame deformed mesh vertices (for rendering/eval), shape (num_frames+1, num_mesh_verts, 3)
        "xt_orig_vertices": xt_orig_vertices,
        # Original mesh face connectivity, shape (num_faces, 3)
        "orig_faces": mesh.faces.cpu(),
        # Material properties at quadrature points
        "sim_yms": so_yms.cpu(),  # Young's moduli, shape (num_qp,)
        "sim_prs": so_prs.cpu(),  # Poisson's ratios, shape (num_qp,)
        "sim_rhos": so_rhos.cpu(),  # Densities, shape (num_qp,)
        # Path to input surface mesh (relative to project root)
        "input_mesh_file": original_mesh_file,
        # Boundary condition target positions per frame, shape (num_frames+1, num_bdry_pts, 3)
        "separate_bdry_pos_seq": bdry_pos_seq.cpu() if bdry_pos_seq is not None else None,
    }
    torch.save(save_dict, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
