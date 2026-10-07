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
Export Thingi10K meshes: normalize, rotate, extract and repair surface mesh,
and generate boundary conditions.

Reads ftetwild tet meshes from datasets/Thingi10K/ftetwild_output_msh/,
processes them, and writes surface meshes to datasets/Thingi10K/processed/{file_id}/.

Boundary condition npz files are generated into config/thingi10k/.

Usage:
    uv run python data/export_thingi10k.py
"""

import os
import numpy as np
import meshio
import igl
import trimesh
import pandas as pd
from boundary_conditions import (save_fix_side, save_pull_farthest_points,
                                 save_pull_boundary)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets", "Thingi10K")
FTETWILD_DIR = os.path.join(DATASETS_DIR, "ftetwild_output_msh")
OUTPUT_BASE_DIR = os.path.join(DATASETS_DIR, "processed")
CONFIG_BASE_DIR = os.path.join(ROOT_DIR, "config", "thingi10k")


def load_tet_msh(file_path):
    mesh = meshio.read(file_path)
    vertices = mesh.points
    tets = None
    for cell in mesh.cells:
        if cell.type == "tetra":
            tets = cell.data
            break
    if tets is None:
        raise ValueError(f"No tetrahedral cells found in {file_path}")
    print(f"  Loaded: {vertices.shape[0]} vertices, {tets.shape[0]} tets")
    return vertices, tets


def make_rotation_matrix(axis, degrees):
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Unknown axis: {axis}")


csv_path = os.path.join(SCRIPT_DIR, "thingi10k_20examples.csv")
csv_df = pd.read_csv(csv_path, sep=r'\s+', dtype=str)

for _, row in csv_df.iterrows():
    file_id = int(row["file_id"])
    rotate_axis = row["rotate_axis"]
    rotate_degree = float(row["rotate_degree"])
    category = row["category"]

    print(f"\n{'='*60}")
    print(f"Processing {file_id} ({category})")

    output_dir = os.path.join(OUTPUT_BASE_DIR, str(file_id))
    os.makedirs(output_dir, exist_ok=True)

    # Load ftetwild tet mesh
    ftetwild_file = os.path.join(FTETWILD_DIR, f"{file_id}.stl_0121.msh")
    if not os.path.exists(ftetwild_file):
        print(f"  WARNING: {ftetwild_file} not found, skipping")
        continue

    ftet_vertices, ftets = load_tet_msh(ftetwild_file)

    # Normalize to [-0.5, 0.5]
    bbox_size = (ftet_vertices.max(axis=0) - ftet_vertices.min(axis=0)).max()
    bbox_center = (ftet_vertices.max(axis=0) + ftet_vertices.min(axis=0)) / 2
    ftet_vertices = (ftet_vertices - bbox_center[None, :]) / bbox_size

    # Apply rotation
    R = make_rotation_matrix(rotate_axis, rotate_degree)
    ftet_vertices = ftet_vertices @ R.T

    # Save normalized tet mesh (needed for FEM simulation)
    tet_mesh = meshio.Mesh(points=ftet_vertices, cells=[("tetra", ftets)])
    tet_mesh_file = os.path.join(output_dir, f"{file_id}.stl_0121.msh")
    meshio.write(tet_mesh_file, tet_mesh, file_format="gmsh22")
    print(f"  Tet mesh -> {tet_mesh_file}")

    # Extract boundary facets and repair surface mesh
    boundary_facets = igl.boundary_facets(ftets)
    if isinstance(boundary_facets, tuple):
        boundary_facets = boundary_facets[0]
    boundary_facets = np.stack([boundary_facets[:, 1], boundary_facets[:, 0], boundary_facets[:, 2]], axis=1)

    surf_v, surf_f, _, _ = igl.remove_unreferenced(ftet_vertices, boundary_facets)
    surf_v, surf_f, _ = igl.split_nonmanifold(surf_v, surf_f)
    surf_f, _ = igl.bfs_orient(surf_f)

    surface_mesh_file = os.path.join(output_dir, f"{file_id}_surface.ply")
    trimesh.Trimesh(vertices=surf_v, faces=surf_f, process=False).export(surface_mesh_file)
    print(f"  Surface mesh: {surf_v.shape[0]} vertices -> {surface_mesh_file}")

    # Boundary conditions. Generated here rather than shipped, so the repo does
    # not redistribute geometry derived from third-party Thingi10K models.
    config_dir = os.path.join(CONFIG_BASE_DIR, str(file_id))
    save_fix_side(ftet_vertices, config_dir, side=row["fix_side"], percentage=5)
    save_pull_farthest_points(ftet_vertices, surf_v, config_dir,
                              num_points=4, offset_percentage=10)
    save_pull_boundary(ftet_vertices, surf_v, config_dir,
                       axis="longest", percentage=5, offset_percentage=10)

print("\nDone.")
