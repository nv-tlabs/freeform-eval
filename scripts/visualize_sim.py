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
Visualize simulation results with polyscope.

Shows deformed mesh frames with a slider, optionally side-by-side with FEM GT.

Usage:
    # Single result:
    DISPLAY=:1 uv run python scripts/visualize_sim.py \
        --pred datasets/NVSimReady/output/{fid}/sim_result_rkpm_model_fix_front_5percent_ym1e5.pth

    # Compare with FEM GT:
    DISPLAY=:1 uv run python scripts/visualize_sim.py \
        --pred datasets/NVSimReady/output/{fid}/sim_result_rkpm_model_fix_front_5percent_ym1e5.pth \
        --gt "datasets/NVSimReady/processed/{fid}/fem_sim_fix_front_5percent_ym1e5/frame_{:04d}.msh"

    # Compare two predictions:
    DISPLAY=:1 uv run python scripts/visualize_sim.py \
        --pred datasets/.../sim_result_rkpm_model_*.pth \
        --pred2 datasets/.../sim_result_mlp_model_*.pth
"""

import argparse
import os
import numpy as np
import torch
import meshio
import igl
import polyscope as ps


def load_pred_frames(path):
    """Load predicted frames from .pth file."""
    data = torch.load(path, weights_only=False)
    if "xt_orig_vertices" in data:
        frames = [x.numpy() if isinstance(x, torch.Tensor) else x
                  for x in data["xt_orig_vertices"]]
        faces = data["orig_faces"].numpy() if isinstance(data["orig_faces"], torch.Tensor) else data["orig_faces"]
    else:
        frames = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                  for x in data["xt_pts"]]
        faces = None
    return frames, faces


def load_gt_frames(gt_format):
    """Load GT frames from msh files."""
    frames = []
    tets = None
    i = 0
    while os.path.exists(gt_format.format(i)):
        msh = meshio.read(gt_format.format(i))
        frames.append(msh.points)
        if tets is None:
            for c in msh.cells:
                if c.type == "tetra":
                    tets = c.data
                    break
        i += 1
    # Extract surface
    boundary = igl.boundary_facets(tets)
    if isinstance(boundary, tuple):
        boundary = boundary[0]
    faces = boundary[:, [1, 0, 2]]
    return frames, faces


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation results")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction .pth")
    parser.add_argument("--gt", type=str, default=None, help="GT format string (e.g. path/frame_{:04d}.msh)")
    parser.add_argument("--pred2", type=str, default=None, help="Second prediction .pth for comparison")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Labels for meshes")
    args = parser.parse_args()

    # Load data
    pred_frames, pred_faces = load_pred_frames(args.pred)
    meshes = [("Prediction", pred_frames, pred_faces)]

    if args.gt:
        gt_frames, gt_faces = load_gt_frames(args.gt)
        meshes.insert(0, ("FEM GT", gt_frames, gt_faces))

    if args.pred2:
        pred2_frames, pred2_faces = load_pred_frames(args.pred2)
        meshes.append(("Prediction 2", pred2_frames, pred2_faces))

    if args.labels:
        for i, label in enumerate(args.labels):
            if i < len(meshes):
                meshes[i] = (label, meshes[i][1], meshes[i][2])

    num_frames = min(len(m[1]) for m in meshes)
    print(f"Loaded {len(meshes)} mesh(es), {num_frames} frames each")

    # Setup polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    # Offset meshes side by side
    offsets = []
    if len(meshes) > 1:
        v0 = meshes[0][1][0]
        bbox_width = (v0.max(axis=0) - v0.min(axis=0)).max() * 1.5
        for i in range(len(meshes)):
            offsets.append(np.array([i * bbox_width, 0, 0]))
    else:
        offsets = [np.array([0, 0, 0])]

    # Register initial meshes
    ps_meshes = []
    for i, (label, frames, faces) in enumerate(meshes):
        v = frames[0].copy()
        v += offsets[i]
        if faces is not None:
            m = ps.register_surface_mesh(label, v, faces, smooth_shade=True)
        else:
            m = ps.register_point_cloud(label, v, radius=0.002)
        ps_meshes.append(m)

    state = {"frame": 0, "playing": False}

    def callback():
        changed, new_frame = ps.imgui.SliderInt("Frame", state["frame"], 0, num_frames - 1)
        if changed:
            state["frame"] = new_frame

        if ps.imgui.Button("Play" if not state["playing"] else "Pause"):
            state["playing"] = not state["playing"]

        ps.imgui.SameLine()
        if ps.imgui.Button("Reset"):
            state["frame"] = 0
            state["playing"] = False

        if state["playing"]:
            state["frame"] = (state["frame"] + 1) % num_frames

        idx = state["frame"]

        for i, (label, frames, faces) in enumerate(meshes):
            v = frames[idx].copy()
            v += offsets[i]
            if faces is not None:
                ps_meshes[i].update_vertex_positions(v)
            else:
                ps_meshes[i].update_point_positions(v)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
