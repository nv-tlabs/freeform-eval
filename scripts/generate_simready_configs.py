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

"""Generate create/sim config files for simready examples."""
import os
import pandas as pd
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT, "data", "simready_20examples.csv")
df = pd.read_csv(csv_path, sep=r'\s+', dtype=str)

# All simready use fix_front, ym=1e5
fix_side = "front"
ym = "1e5"

for _, row in df.iterrows():
    fid = row["file_id"]
    desc = row["description"]
    cfg_dir = os.path.join(ROOT, "config", "simready", fid)
    tet_npz_dir = os.path.join(ROOT, "datasets", "NVSimReady", "tet_npz", fid)

    if not os.path.isdir(tet_npz_dir):
        print(f"SKIP {desc}: no tet_npz dir")
        continue

    # Find mesh and materials files
    msh_files = [f for f in os.listdir(tet_npz_dir) if f.endswith('.msh')]
    mat_files = [f for f in os.listdir(tet_npz_dir) if f.endswith('_materials.npz')]
    if not msh_files:
        print(f"SKIP {desc}: no .msh")
        continue

    mesh_name = os.path.splitext(msh_files[0])[0]
    msh_rel = f"datasets/NVSimReady/tet_npz/{fid}/{msh_files[0]}"
    surface_rel = f"datasets/NVSimReady/processed/{fid}/{mesh_name}_surface.ply"
    mat_rel = f"datasets/NVSimReady/tet_npz/{fid}/{mat_files[0]}" if mat_files else None

    bc_names = [
        f"fix_{fix_side}_5percent",
        "pull_farthest_points_4_10percent",
        "pull_boundary_longest_5percent_offset_10percent",
    ]

    # --- create_rkpm.yaml ---
    rkpm_cfg = {
        "method": "rkpm",
        "surface_mesh": surface_rel,
        "output_dir": f"datasets/NVSimReady/output/{fid}",
        "young_modulus": float(ym),
        "poisson_ratio": 0.45,
        "density": 500.0,
        # num_handles: new kaolin subtracts 1, so 33 gives 32 eigenmodes + 1 constant
        "num_handles": 33,
        "num_nodes": 1000,
        "num_sample_pts": 262144,  # 64^3
    }
    if mat_rel:
        rkpm_cfg["materials"] = mat_rel
    OmegaConf.save(OmegaConf.create(rkpm_cfg), os.path.join(cfg_dir, "create_rkpm.yaml"))

    # --- create_mlp.yaml ---
    mlp_cfg = {
        "method": "mlp",
        "surface_mesh": surface_rel,
        "output_dir": f"datasets/NVSimReady/output/{fid}",
        "young_modulus": float(ym),
        "poisson_ratio": 0.45,
        "density": 500.0,
        # num_handles: new kaolin subtracts 1, so 33 gives 32 learned + 1 constant
        "num_handles": 33,
        "num_layers": 6,
        "num_sample_pts": 1000000,
        "sample_method": "random_volume",
        "num_training_samples": 1000,
        "num_steps": 10000,
        "lr_start": 1e-3,
        "lr_end": 1e-3,
        "le_coeff": 1e-1,
        "lo_coeff": 1e6,
    }
    if mat_rel:
        mlp_cfg["materials"] = mat_rel
    OmegaConf.save(OmegaConf.create(mlp_cfg), os.path.join(cfg_dir, "create_mlp.yaml"))

    # --- sim configs ---
    for bc_name in bc_names:
        bc_yaml = f"{bc_name}_ym{ym}"
        fem_cfg_path = os.path.join(cfg_dir, f"{bc_yaml}.yaml")
        if os.path.exists(fem_cfg_path):
            fem_cfg = OmegaConf.load(fem_cfg_path)
            dt = fem_cfg.get("dt", 0.05)
            num_steps = fem_cfg.get("n_frames", 100)
        else:
            dt = 0.05
            num_steps = 100
        max_newton = 20

        for method in ["rkpm", "mlp"]:
            sim_cfg = {
                "model": f"datasets/NVSimReady/output/{fid}/{method}_model.pth",
                "boundary_config": f"config/simready/{fid}/{bc_yaml}.yaml",
                "output_dir": f"datasets/NVSimReady/output/{fid}",
                "separate_bdry_weight": 1e9,  # simready uses 1e9
                "num_samples": 125000,
                "num_steps": num_steps,
                "dt": dt,
                "max_newton_steps": max_newton,
                "apply_qr": True,
                "truncate_qr_rank": True,
            }
            OmegaConf.save(OmegaConf.create(sim_cfg),
                           os.path.join(cfg_dir, f"sim_{method}_{bc_name}.yaml"))

    print(f"Generated: {desc} ({fid[:12]}...)")

print(f"\nDone. Generated configs for {len(df)} examples.")
