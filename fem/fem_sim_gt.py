# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# This file is derived from VoMP's simulation/warp.fem/drop_tetmesh.py
# (https://github.com/nv-tlabs/VoMP) and has been modified. Changes include:
# - Boundary conditions driven by a config file and a precomputed boundary
#   position sequence, rather than hardcoded clamped edges
# - Spatially varying materials applied from a VoMP materials .npz
# - Per-step displacement export for evaluation against reduced-order results

import argparse
import os
import numpy as np
from omegaconf import OmegaConf

import meshio
import warp as wp
import warp.fem as fem
from warp.fem import Domain, Sample, Field
from warp.fem import integrand

import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
# Add VoMP's warp.fem to path for base mfem modules
_vomp_fem = os.path.join(_project_root, "third_party", "VoMP", "simulation", "warp.fem")
sys.path.insert(0, _vomp_fem)
from mfem.softbody_sim import ClassicFEM, run_softbody_sim

import trimesh


@integrand
def clamped_boundary(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
    boundary_indices: wp.array(dtype=int),
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""
    clamped = float(0.0)
    for i in range(boundary_indices.shape[0]):
        if s.qp_index == boundary_indices[i]:
            clamped = 1.0
    return wp.dot(u(s), v(s)) * clamped


@integrand
def boundary_displacement_form(
    s: Sample,
    domain: Domain,
    v: Field,
    boundary_indices: wp.array(dtype=int),
    boundary_pos_delta: wp.array(dtype=wp.vec3f),
):
    """Prescribed displacement"""
    for i in range(boundary_indices.shape[0]):
        if s.qp_index == boundary_indices[i]:
            return wp.dot(boundary_pos_delta[i], v(s))


if __name__ == "__main__":
    wp.init()

    sim_class = ClassicFEM
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-vol-results", action="store_true")
    parser.add_argument("--save-surface-results", action="store_true")
    sim_class.add_parser_arguments(parser)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_file)

    args.neo_hookean = True
    args.n_frames = config["n_frames"]
    args.dt = config["dt"]
    args.gravity = config["gravity"]
    args.n_newton = config["n_newton"]
    args.newton_tol = config["newton_tol"]
    args.ground = config.get("ground", False)
    args.ground_height = config.get("ground_height", 0.0)

    # Config paths are relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    msh_file = os.path.join(project_root, config["msh_file"])
    young_modulus = config.get("young_modulus", None)
    if young_modulus is not None:
        args.young_modulus = young_modulus
    poisson_ratio = config.get("poisson_ratio", None)
    if poisson_ratio is not None:
        args.poisson_ratio = poisson_ratio
    density = config.get("density", None)
    if density is not None:
        args.density = density
    print(args)

    msh = meshio.read(msh_file)
    pos = wp.array(msh.points, dtype=wp.vec3f)
    assert msh.cells[0].type == "tetra"
    tets = wp.array(msh.cells[0].data, dtype=wp.int32)
    pos.requires_grad = True
    geo = fem.Tetmesh(positions=pos, tet_vertex_indices=tets, build_bvh=True)
    vtx_quadrature = fem.PicQuadrature(fem.Cells(geo), pos)

    sim = sim_class(geo, active_cells=None, args=args)
    sim.init_displacement_space(None)
    sim.init_strain_spaces()

    if "materials" in config:
        from fem.material_loader import apply_spatially_varying_materials
        materials_path = os.path.join(project_root, config["materials"])
        print(f"Applying spatially varying materials from {materials_path}")
        apply_spatially_varying_materials(sim, materials_path, k_neighbors=1)

    has_boundary_file = config.get("boundary_file", None) is not None
    if has_boundary_file:
        boundary_file = os.path.join(project_root, config["boundary_file"])
        boundary_data = np.load(boundary_file)
        boundary_indices = boundary_data["boundary_indices"]
        boundary_pos_seq = boundary_data["boundary_pos_seq"]

        wp_boundary_pos_seq = wp.array(boundary_pos_seq, dtype=wp.vec3f)
        assert boundary_pos_seq.shape[0] == args.n_frames + 1

        wp_boundary_indices = wp.array(boundary_indices, dtype=int)
        wp_boundary_pos_delta = wp_boundary_pos_seq[1] - wp_boundary_pos_seq[0]
    else:
        wp_boundary_indices = wp.zeros(0, dtype=int)
        wp_boundary_pos_delta = wp.zeros(0, dtype=wp.vec3f)

    boundary_projector_form = clamped_boundary
    sim.set_boundary_condition(
        boundary_projector_form=boundary_projector_form,
        boundary_projector_args={
            "boundary_indices": wp_boundary_indices,
        },
        boundary_displacement_form=boundary_displacement_form,
        boundary_displacement_args={
            "boundary_indices": wp_boundary_indices,
            "boundary_pos_delta": wp_boundary_pos_delta,
        },
    )

    def save_vol_result(frame_idx):
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else os.path.join(project_root, config["base_dir"])
        )
        config_name = os.path.basename(args.config_file).split(".")[0]
        vol_output_dir = os.path.join(output_dir, "fem_sim_" + config_name)
        os.makedirs(vol_output_dir, exist_ok=True)
        tets = sim.u_field.space.node_tets()
        active_indices = sim.u_field.space_partition.space_node_indices().numpy()
        displaced_pos = sim.u_field.space.node_positions().numpy()
        displaced_pos[active_indices] += sim.u_field.dof_values.numpy()
        output_file = os.path.join(vol_output_dir, f"frame_{frame_idx:04d}.msh")
        mesh = meshio.Mesh(
            points=displaced_pos.astype(float), cells=[("tetra", tets.astype(int))]
        )
        meshio.write(output_file, mesh, file_format="gmsh22")

    def save_surface_result(frame_idx):
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else os.path.join(project_root, config["base_dir"])
        )
        config_name = os.path.basename(args.config_file).split(".")[0]
        surface_output_dir = os.path.join(output_dir, "fem_sim_surface_" + config_name)
        os.makedirs(surface_output_dir, exist_ok=True)
        faces = sim.geo._face_vertex_indices.numpy()
        bd_faces = faces[sim.geo._boundary_face_indices.numpy()]
        surface_pos = sim.geo.positions.numpy()
        surface_pos += sim.u_field.dof_values.numpy()[: surface_pos.shape[0]]
        output_file = os.path.join(surface_output_dir, f"frame_{frame_idx:04d}.ply")
        trimesh.Trimesh(surface_pos, bd_faces, process=False).export(output_file)

    def init_callback():
        if args.ui and has_boundary_file:
            import polyscope as ps
            sim.ps_boundary_points = ps.register_point_cloud("boundary points", wp_boundary_pos_seq[0].numpy(), enabled=True)
        if args.save_vol_results:
            save_vol_result(0)
        if args.save_surface_results:
            save_surface_result(0)

    def frame_callback(pos):
        frame_idx = sim.cur_frame
        if has_boundary_file:
            if frame_idx < args.n_frames:
                print(f"Updating boundary pos for frame {frame_idx} -> {frame_idx + 1}")
                wp_boundary_pos_delta.assign(
                    wp_boundary_pos_seq[frame_idx + 1] - wp_boundary_pos_seq[frame_idx]
                )
                sim.set_boundary_condition(
                    boundary_projector_form=boundary_projector_form,
                    boundary_projector_args={
                        "boundary_indices": wp_boundary_indices,
                    },
                    boundary_displacement_form=boundary_displacement_form,
                    boundary_displacement_args={
                        "boundary_indices": wp_boundary_indices,
                        "boundary_pos_delta": wp_boundary_pos_delta,
                    },
                )
            if args.ui:
                sim.ps_boundary_points.update_point_positions(wp_boundary_pos_seq[frame_idx].numpy())
        if args.save_vol_results:
            save_vol_result(frame_idx)
        if args.save_surface_results:
            save_surface_result(frame_idx)

    with wp.ScopedTimer("run_softbody_sim", active=True, use_nvtx=True) as timer:
        run_softbody_sim(
            sim,
            ui=args.ui,
            init_callback=init_callback,
            frame_callback=frame_callback,
        )
