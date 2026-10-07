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
Compute residual error: how well can the skinning weights basis represent GT deformations.

For each GT frame, projects the displacement onto the LBS basis via least-squares,
then measures the residual (reconstruction error). This quantifies the representation
capacity of the basis independent of the simulation solver.

Usage:
    uv run python eval/compute_residual_error.py \
        --gt-path "datasets/Thingi10K/processed/96123/fem_sim_fix_front_5percent_ym1e4/frame_{:04d}.msh" \
        --model-path datasets/Thingi10K/output/96123/rkpm_model.pth
"""

import os
import argparse
import numpy as np
import torch
import pickle
import time
import json
import meshio

import kaolin.physics.simplicits


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
    return vertices, tets


def load_ground_truth(gt_format, force_reload=False):
    cache_file = os.path.splitext(gt_format.format(0))[0] + ".pkl"

    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading GT from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            gt_verts, gt_tets = pickle.load(f)
    else:
        print(f"Loading GT from: {gt_format}")
        gt_verts = []
        gt_tets = None
        num_frames = 0
        while os.path.exists(gt_format.format(num_frames)):
            num_frames += 1
        for i in range(num_frames):
            verts, tets = load_tet_msh(gt_format.format(i))
            if gt_tets is None:
                gt_tets = tets
            gt_verts.append(verts)
        with open(cache_file, "wb") as f:
            pickle.dump((gt_verts, gt_tets), f)

    print(f"Loaded {len(gt_verts)} GT frames, {gt_verts[0].shape[0]} vertices")
    return gt_verts, gt_tets


def main():
    parser = argparse.ArgumentParser(description='Compute residual error')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Format string for GT files (e.g., path/frame_{:04d}.msh)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model .pth file (rkpm_model.pth or mlp_model.pth)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--force-reload', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    gt_verts, gt_tets = load_ground_truth(args.gt_path, force_reload=args.force_reload)

    device = torch.device('cuda')
    saved_model = torch.load(args.model_path, weights_only=False)
    so_model = saved_model['model']
    so_model.to(device=device)

    # Rest pose from first GT frame
    rest_verts = torch.from_numpy(gt_verts[0]).to(device=device, dtype=torch.float32)

    # Build LBS matrix from skinning weights (non-truncated)
    weights = so_model.compute_skinning_weights(rest_verts)
    B = kaolin.physics.simplicits.lbs_matrix(rest_verts, weights).detach()

    # Bbox normalization
    gt_min = gt_verts[0].min(axis=0)
    gt_max = gt_verts[0].max(axis=0)
    max_bbox_size = float((gt_max - gt_min).max())

    num_frames = len(gt_verts)
    frame_errors = []
    for i in range(num_frames):
        x = torch.from_numpy(gt_verts[i]).to(device=device, dtype=torch.float32)
        dx = x - rest_verts
        results = torch.linalg.lstsq(B, dx.view(-1, 1))
        z = results.solution
        residual = torch.sum((dx - (B @ z).view(-1, 3)) ** 2, dim=-1).mean().item()
        frame_errors.append(residual / max_bbox_size ** 2)

    frame_errors = np.array(frame_errors)
    mean_error = float(frame_errors.mean())
    std_error = float(frame_errors.std())

    print(f"\nNormalized MSE (residual): {mean_error:.6e}")
    print(f"Normalized RMSE (residual): {np.sqrt(mean_error):.6e}")
    print(f"Std: {std_error:.6e}")

    gt_name = os.path.basename(os.path.dirname(args.gt_path))
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]

    results = {
        'gt_path': args.gt_path,
        'model_path': args.model_path,
        'max_bbox_size': max_bbox_size,
        'num_frames': int(num_frames),
        'normalized_mean_squared_error': mean_error,
        'normalized_std_error': std_error,
        'normalized_rmse': float(np.sqrt(mean_error)),
        'normalized_min_error': float(frame_errors.min()),
        'normalized_max_error': float(frame_errors.max()),
        'normalized_frame_errors': frame_errors.tolist(),
    }

    json_path = os.path.join(args.output_dir, f"{model_basename}_{gt_name}_residual_error_stats.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {json_path}")


if __name__ == "__main__":
    main()
