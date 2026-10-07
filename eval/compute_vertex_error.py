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
Compute vertex error statistics between FEM ground truth and predicted simulation.

Usage:
    uv run python eval/compute_vertex_error.py \
        --gt-path datasets/Thingi10K/processed/96123/fem_sim_fix_front_5percent_ym1e4/frame_{:04d}.msh \
        --pred-path output/thingi10k/96123/sim_result_fix_front_5percent_ym1e4.pth
"""

import os
import argparse
import numpy as np
import torch
import pickle
import time
import json
import meshio


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


def load_predicted_results(pred_path):
    print(f"Loading predictions from: {pred_path}")
    sim_result = torch.load(pred_path, weights_only=False)
    xt_pts = [x.detach().cpu().numpy().astype(np.float64) if isinstance(x, torch.Tensor) else x
              for x in sim_result['xt_pts']]
    print(f"Loaded {len(xt_pts)} predicted frames, {xt_pts[0].shape[0]} vertices")
    return xt_pts


def compute_vertex_errors(gt_verts, gt_tets, xt_pts):
    import igl

    gt_min = gt_verts[0].min(axis=0)
    gt_max = gt_verts[0].max(axis=0)
    max_bbox_size = (gt_max - gt_min).max()
    print(f"Max bbox dimension: {max_bbox_size:.6f}")

    # Barycentric mapping from prediction points to GT tets
    D, I, C = igl.point_mesh_squared_distance(xt_pts[0], gt_verts[0], gt_tets)
    bary_coords = igl.barycentric_coordinates(
        xt_pts[0],
        gt_verts[0][gt_tets[I, 0]],
        gt_verts[0][gt_tets[I, 1]],
        gt_verts[0][gt_tets[I, 2]],
        gt_verts[0][gt_tets[I, 3]],
    )

    first_gt_interp = (bary_coords[:, :, None] * gt_verts[0][gt_tets[I]]).sum(axis=1)
    final_gt_interp = (bary_coords[:, :, None] * gt_verts[-1][gt_tets[I]]).sum(axis=1)
    final_disp_sq = ((final_gt_interp - first_gt_interp) ** 2).sum(axis=-1).mean()

    num_frames = min(len(gt_verts), len(xt_pts))
    frame_errors = []
    relative_frame_errors = []

    for i in range(num_frames):
        gt_interp = (bary_coords[:, :, None] * gt_verts[i][gt_tets[I]]).sum(axis=1)
        error = xt_pts[i] - gt_interp
        normalized_error = error / max_bbox_size
        frame_mse = (normalized_error ** 2).sum(axis=-1).mean()
        frame_errors.append(frame_mse)
        rel_err = ((xt_pts[i] - gt_interp) ** 2).sum(axis=-1).mean() / final_disp_sq
        relative_frame_errors.append(rel_err)

    return np.array(frame_errors), np.array(relative_frame_errors), max_bbox_size


def main():
    parser = argparse.ArgumentParser(description='Compute vertex error statistics')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Format string for GT files (e.g., path/frame_{:04d}.msh)')
    parser.add_argument('--pred-path', type=str, required=True,
                        help='Path to prediction .pth file')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--force-reload', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pred_path)
    os.makedirs(args.output_dir, exist_ok=True)

    gt_verts, gt_tets = load_ground_truth(args.gt_path, force_reload=args.force_reload)
    xt_pts = load_predicted_results(args.pred_path)

    frame_errors, relative_frame_errors, max_bbox_size = compute_vertex_errors(gt_verts, gt_tets, xt_pts)

    mean_mse = frame_errors.mean()
    print(f"\nNormalized MSE: {mean_mse:.6e}")
    print(f"Normalized RMSE: {np.sqrt(mean_mse):.6e}")
    print(f"Relative MSE: {relative_frame_errors.mean():.6e}")
    print(f"Max frame error: {frame_errors.max():.6e}")

    results = {
        'gt_path': args.gt_path,
        'pred_path': args.pred_path,
        'max_bbox_size': float(max_bbox_size),
        'num_frames': int(len(frame_errors)),
        'normalized_mean_squared_error': float(mean_mse),
        'normalized_rmse': float(np.sqrt(mean_mse)),
        'normalized_frame_errors': frame_errors.tolist(),
        'relative_frame_errors': relative_frame_errors.tolist(),
    }

    pred_basename = os.path.splitext(os.path.basename(args.pred_path))[0]
    json_path = os.path.join(args.output_dir, f"{pred_basename}_vertex_error_stats.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {json_path}")


if __name__ == "__main__":
    main()
