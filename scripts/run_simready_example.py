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
Run full pipeline for one simready example.

Usage:
    uv run python scripts/run_simready_example.py --fid 014841555b1c1ce0590f43ebe4863e5415a526b5afba642e459f48744c4bd05a
"""

import argparse
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# All simready use these
FIX_SIDE = "front"
YM = "1e5"


def run(cmd, label):
    print(f"  [{label}] {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [{label}] FAILED ({elapsed:.1f}s)")
        print(result.stderr[-500:] if result.stderr else "")
        return False
    print(f"  [{label}] done ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=str, required=True)
    parser.add_argument("--residual-error", action="store_true", help="Also compute residual error")
    args = parser.parse_args()

    fid = args.fid
    cfg = f"config/simready/{fid}"
    bcs = [
        f"fix_{FIX_SIDE}_5percent",
        "pull_farthest_points_4_10percent",
        "pull_boundary_longest_5percent_offset_10percent",
    ]

    # 1. FEM GT
    for bc in bcs:
        bc_ym = f"{bc}_ym{YM}"
        gt_dir = f"datasets/NVSimReady/processed/{fid}/fem_sim_{bc_ym}"
        if os.path.isdir(gt_dir) and len([f for f in os.listdir(gt_dir) if f.endswith('.msh')]) > 50:
            print(f"  [FEM] {bc_ym} exists, skip")
        else:
            run(f"uv run python fem/fem_sim_gt.py --config-file {cfg}/{bc_ym}.yaml "
                f"--save-vol-results --cg_iters 10000 --cg_tol 1e-8 --fp64 --no-ui",
                f"FEM {bc_ym}")

    # 2. RKPM
    if os.path.exists(f"datasets/NVSimReady/output/{fid}/rkpm_model.pth"):
        print(f"  [RKPM] model exists, skip")
    else:
        run(f"uv run python sim/create_model.py --config {cfg}/create_rkpm.yaml", "RKPM create")

    # 3. MLP
    if os.path.exists(f"datasets/NVSimReady/output/{fid}/mlp_model.pth"):
        print(f"  [MLP] model exists, skip")
    else:
        run(f"uv run python sim/create_model.py --config {cfg}/create_mlp.yaml", "MLP create")

    # 4. Sim + Eval
    for method in ["rkpm", "mlp"]:
        for bc in bcs:
            bc_ym = f"{bc}_ym{YM}"
            pred = f"datasets/NVSimReady/output/{fid}/sim_result_{method}_model_{bc_ym}.pth"
            gt_fmt = f"datasets/NVSimReady/processed/{fid}/fem_sim_{bc_ym}/frame_{{:04d}}.msh"
            eval_json = pred.replace(".pth", "_vertex_error_stats.json")

            if os.path.exists(pred):
                print(f"  [SIM] {method.upper()} {bc} exists, skip")
            else:
                run(f"uv run python sim/run_sim.py --config {cfg}/sim_{method}_{bc}.yaml",
                    f"SIM {method.upper()} {bc}")

            if os.path.exists(eval_json):
                print(f"  [EVAL] {method.upper()} {bc} exists, skip")
            else:
                run(f'uv run python eval/compute_vertex_error.py --gt-path "{gt_fmt}" --pred-path "{pred}"',
                    f"EVAL {method.upper()} {bc}")

    # 5. Residual error (basis representation capacity, optional)
    if not args.residual_error:
        print(f"  [DONE] {fid}")
        return
    for method in ["rkpm", "mlp"]:
        model_path = f"datasets/NVSimReady/output/{fid}/{method}_model.pth"
        for bc in bcs:
            bc_ym = f"{bc}_ym{YM}"
            gt_fmt = f"datasets/NVSimReady/processed/{fid}/fem_sim_{bc_ym}/frame_{{:04d}}.msh"
            resid_json = f"datasets/NVSimReady/output/{fid}/{method}_model_fem_sim_{bc_ym}_residual_error_stats.json"

            if os.path.exists(resid_json):
                print(f"  [RESID] {method.upper()} {bc} exists, skip")
            else:
                run(f'uv run python eval/compute_residual_error.py --gt-path "{gt_fmt}" --model-path "{model_path}"',
                    f"RESID {method.upper()} {bc}")

    print(f"  [DONE] {fid}")


if __name__ == "__main__":
    main()
