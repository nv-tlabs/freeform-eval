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

"""Generate all config files for thingi10k examples from the CSV."""
import os
import pandas as pd
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

csv_path = os.path.join(ROOT, "data", "thingi10k_20examples.csv")
df = pd.read_csv(csv_path, sep=r'\s+', dtype=str)

BCS = ["fix_{side}_{pct}percent", "pull_farthest_points_4_10percent",
       "pull_boundary_longest_5percent_offset_10percent"]

for _, row in df.iterrows():
    fid = row["file_id"]
    ym = row["yms"]
    fix_side = row["fix_side"]
    cfg_dir = os.path.join(ROOT, "config", "thingi10k", fid)
    os.makedirs(cfg_dir, exist_ok=True)

    # Boundary condition names for this example
    bc_names = [
        f"fix_{fix_side}_5percent",
        "pull_farthest_points_4_10percent",
        "pull_boundary_longest_5percent_offset_10percent",
    ]

    # --- create_rkpm.yaml ---
    rkpm_cfg = {
        "method": "rkpm",
        "surface_mesh": f"datasets/Thingi10K/processed/{fid}/{fid}_surface.ply",
        "output_dir": f"datasets/Thingi10K/output/{fid}",
        "young_modulus": float(ym),
        "poisson_ratio": 0.45,
        "density": 500.0,
        # num_handles: new kaolin subtracts 1 internally, so 33 gives 32 eigenmodes + 1 constant
        "num_handles": 33,
        "num_nodes": 1000,
        "num_sample_pts": 262144,  # 64^3
    }
    OmegaConf.save(OmegaConf.create(rkpm_cfg),
                    os.path.join(cfg_dir, "create_rkpm.yaml"))

    # --- create_mlp.yaml ---
    mlp_cfg = {
        "method": "mlp",
        "surface_mesh": f"datasets/Thingi10K/processed/{fid}/{fid}_surface.ply",
        "output_dir": f"datasets/Thingi10K/output/{fid}",
        "young_modulus": float(ym),
        "poisson_ratio": 0.45,
        "density": 500.0,
        # num_handles: new kaolin subtracts 1 internally, so 33 gives 32 learned + 1 constant
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
    OmegaConf.save(OmegaConf.create(mlp_cfg),
                    os.path.join(cfg_dir, "create_mlp.yaml"))

    # --- sim configs (per BC, per method) ---
    for bc_name in bc_names:
        bc_yaml = f"{bc_name}_ym{ym}"
        bdry_weight = float(ym)  # boundary weight = YM for thingi10k

        # Read FEM config to get dt and num_steps
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
                "model": f"datasets/Thingi10K/output/{fid}/{method}_model.pth",
                "boundary_config": f"config/thingi10k/{fid}/{bc_yaml}.yaml",
                "output_dir": f"datasets/Thingi10K/output/{fid}",
                "separate_bdry_weight": bdry_weight,
                "num_samples": 125000,
                "num_steps": num_steps,
                "dt": dt,
                "max_newton_steps": max_newton,
                "apply_qr": True,
                "truncate_qr_rank": True,
            }
            # Short BC name for filename
            bc_short = bc_name.replace("_5percent", "").replace("_10percent", "").replace("_longest_offset", "")
            OmegaConf.save(OmegaConf.create(sim_cfg),
                            os.path.join(cfg_dir, f"sim_{method}_{bc_name}.yaml"))

    print(f"Generated configs for {fid}")

print(f"\nDone. Generated configs for {len(df)} examples.")
