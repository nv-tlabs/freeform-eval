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
Export SimReady tet meshes: extract surface mesh from tet mesh.

Reads tet meshes from datasets/NVSimReady/tet_npz/{file_id}/,
extracts and repairs surface mesh, saves to datasets/NVSimReady/processed/{file_id}/.

Usage:
    uv run python data/export_simready.py
"""

import os
import numpy as np
import meshio
import igl
import trimesh
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TET_NPZ_DIR = os.path.join(ROOT_DIR, "datasets", "NVSimReady", "tet_npz")
OUTPUT_DIR = os.path.join(ROOT_DIR, "datasets", "NVSimReady", "processed")

# SimReady boundary conditions ship in config/simready/ rather than being
# generated. The SimReady assets are NVIDIA-owned, so there is no third-party
# redistribution concern, and TetWild's surface output that the shipped
# boundaries were built from is not included here, so they cannot be
# regenerated faithfully. data/boundary_conditions.py has the generators if
# that ever changes.

csv_path = os.path.join(SCRIPT_DIR, "simready_20examples.csv")
df = pd.read_csv(csv_path, sep=r'\s+', dtype=str)

for _, row in df.iterrows():
    fid = row["file_id"]
    desc = row["description"]

    print(f"\n{'='*60}")
    print(f"Processing {desc} ({fid[:12]}...)")

    tet_dir = os.path.join(TET_NPZ_DIR, fid)
    if not os.path.isdir(tet_dir):
        print(f"  WARNING: {tet_dir} not found, skipping")
        continue

    # Find tet mesh
    msh_files = [f for f in os.listdir(tet_dir) if f.endswith('.msh')]
    if not msh_files:
        print(f"  WARNING: no .msh file found, skipping")
        continue
    msh_file = os.path.join(tet_dir, msh_files[0])
    mesh_name = os.path.splitext(msh_files[0])[0]

    # Load tet mesh
    msh = meshio.read(msh_file)
    vertices = msh.points
    tets = None
    for cell in msh.cells:
        if cell.type == "tetra":
            tets = cell.data
            break
    if tets is None:
        print(f"  WARNING: no tetra cells, skipping")
        continue
    print(f"  Tet mesh: {vertices.shape[0]} vertices, {tets.shape[0]} tets")

    # Extract and repair surface mesh
    boundary_facets = igl.boundary_facets(tets)
    if isinstance(boundary_facets, tuple):
        boundary_facets = boundary_facets[0]
    boundary_facets = np.stack([boundary_facets[:, 1], boundary_facets[:, 0], boundary_facets[:, 2]], axis=1)

    surf_v, surf_f, _, _ = igl.remove_unreferenced(vertices, boundary_facets)
    surf_v, surf_f, _ = igl.split_nonmanifold(surf_v, surf_f)
    surf_f, _ = igl.bfs_orient(surf_f)

    out_dir = os.path.join(OUTPUT_DIR, fid)
    os.makedirs(out_dir, exist_ok=True)
    surface_file = os.path.join(out_dir, f"{mesh_name}_surface.ply")
    trimesh.Trimesh(vertices=surf_v, faces=surf_f, process=False).export(surface_file)
    print(f"  Surface mesh: {surf_v.shape[0]} vertices -> {surface_file}")

print("\nDone.")
