#!/bin/bash
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

# Run full beam experiment pipeline.
# Usage:
#   bash scripts/run_beam.sh                    # default (m=32)
#   bash scripts/run_beam.sh --handle-sweep     # run m=6,9,16,32
set -e
cd "$(dirname "$0")/.."

HANDLE_SWEEP=false
for arg in "$@"; do
    if [ "$arg" = "--handle-sweep" ]; then
        HANDLE_SWEEP=true
    fi
done

CFG=config/beam/beam5m_res50x10x10
OUT=datasets/Beam/output/beam5m_res50x10x10
GT_DIR=datasets/Beam/processed/beam5m_res50x10x10

# FEM ground truth (skip if already exists)
for BC_YM in fix_right_0p5m_ym5e6 twist_right_ym5e6; do
    if [ -d "${GT_DIR}/fem_sim_${BC_YM}" ] && [ "$(ls ${GT_DIR}/fem_sim_${BC_YM}/*.msh 2>/dev/null | wc -l)" -gt 50 ]; then
        echo "  [FEM] ${BC_YM} exists, skip"
    else
        echo "  [FEM] ${BC_YM}"
        uv run python fem/fem_sim_gt.py --config-file ${CFG}/${BC_YM}.yaml \
            --save-vol-results --cg_iters 10000 --cg_tol 1e-8 --fp64 --no-ui
    fi
done

if [ "$HANDLE_SWEEP" = true ]; then
    HANDLES="6 9 16 32"
else
    HANDLES="32"
fi

for M in $HANDLES; do
    NH=$((M + 1))  # kaolin subtracts 1 internally

    echo ""
    echo "============================================================"
    echo "  Beam experiments (m=$M handles)"
    echo "============================================================"

    if [ "$M" = "32" ]; then
        SUFFIX=""
    else
        SUFFIX="_h${M}"
    fi

    # --- RKPM ---
    RKPM_MODEL="${OUT}/rkpm${SUFFIX}_model.pth"
    if [ -f "$RKPM_MODEL" ]; then
        echo "  [RKPM] model exists, skip"
    else
        uv run python sim/create_model.py --config ${CFG}/create_rkpm.yaml \
            --override num_handles=$NH output_name="rkpm${SUFFIX}_model.pth"
    fi

    for BC in fix_right_0p5m twist_right; do
        BC_YM="${BC}_ym5e6"
        PRED="${OUT}/sim_result_rkpm${SUFFIX}_model_${BC_YM}.pth"
        EVAL_JSON="${PRED%.pth}_vertex_error_stats.json"

        echo "--- RKPM${SUFFIX} ${BC} ---"
        if [ -f "$PRED" ]; then
            echo "  [SIM] exists, skip"
        else
            uv run python sim/run_sim.py --config ${CFG}/sim_rkpm_${BC}.yaml \
                --override model="$RKPM_MODEL"
        fi

        if [ -f "$EVAL_JSON" ]; then
            echo "  [EVAL] exists, skip"
        else
            uv run python eval/compute_vertex_error.py \
                --gt-path "${GT_DIR}/fem_sim_${BC_YM}/frame_{:04d}.msh" \
                --pred-path "$PRED"
        fi
    done

    # --- MLP ---
    MLP_MODEL="${OUT}/mlp${SUFFIX}_model.pth"
    if [ -f "$MLP_MODEL" ]; then
        echo "  [MLP] model exists, skip"
    else
        uv run python sim/create_model.py --config ${CFG}/create_mlp.yaml \
            --override num_handles=$NH output_name="mlp${SUFFIX}_model.pth"
    fi

    for BC in fix_right_0p5m twist_right; do
        BC_YM="${BC}_ym5e6"
        PRED="${OUT}/sim_result_mlp${SUFFIX}_model_${BC_YM}.pth"
        EVAL_JSON="${PRED%.pth}_vertex_error_stats.json"

        echo "--- MLP${SUFFIX} ${BC} ---"
        if [ -f "$PRED" ]; then
            echo "  [SIM] exists, skip"
        else
            uv run python sim/run_sim.py --config ${CFG}/sim_mlp_${BC}.yaml \
                --override model="$MLP_MODEL"
        fi

        if [ -f "$EVAL_JSON" ]; then
            echo "  [EVAL] exists, skip"
        else
            uv run python eval/compute_vertex_error.py \
                --gt-path "${GT_DIR}/fem_sim_${BC_YM}/frame_{:04d}.msh" \
                --pred-path "$PRED"
        fi
    done
done

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
uv run python eval/summarize_errors.py --dataset beam
