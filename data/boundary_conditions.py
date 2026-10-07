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

"""Boundary condition generation shared by the Thingi10K and SimReady exporters.

Each boundary condition is written as an .npz holding the indices of the
constrained vertices on the volume mesh, their rest positions, and their
prescribed positions over time. These are generated rather than shipped so the
repository does not redistribute geometry derived from third-party models.
"""

import os
import numpy as np
from scipy.spatial import cKDTree

NUM_FRAMES = 100


def farthest_point_sampling(points, k):
    """Greedy farthest-point sampling seeded at index 0.

    Matches pytorch3d.ops.sample_farthest_points with random_start_point=False:
    take point 0, then repeatedly take the point whose squared distance to the
    already-selected set is largest. np.argmax breaks ties toward the lower
    index, as torch.argmax does.

    Note this is seeded by array position, so the selected points depend on the
    vertex ordering of the surface mesh. libigl is pinned in the README for
    that reason.
    """
    selected = np.zeros(k, dtype=np.int64)
    closest = ((points - points[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        idx = int(np.argmax(closest))
        selected[i] = idx
        closest = np.minimum(closest, ((points - points[idx]) ** 2).sum(axis=1))
    return selected


def save_fix_side(ftet_vertices, config_dir, side="front", percentage=5):
    """Pin the vertices within `percentage` of one end of an axis.

    Selection and positions come from the tet mesh rather than the surface mesh,
    so this is independent of the surface repair.
    """
    seq_name = f"fix_{side}_{percentage}percent"
    if side in ("front", "back"):
        axis_idx, take_max = 2, side == "back"
    elif side in ("top", "bottom"):
        axis_idx, take_max = 1, side == "top"
    elif side in ("left", "right"):
        axis_idx, take_max = 0, side == "right"
    else:
        raise ValueError(f"Invalid side: {side}")

    ratio = percentage / 100.0
    sorted_indices = np.argsort(ftet_vertices[:, axis_idx])
    num_vertices = ftet_vertices.shape[0]
    if take_max:
        cutoff = ftet_vertices[sorted_indices[-int(num_vertices * ratio) + 1], axis_idx]
        boundary_indices = np.where(ftet_vertices[:, axis_idx] >= -1e-3 + cutoff)[0]
    else:
        cutoff = ftet_vertices[sorted_indices[int(num_vertices * ratio)], axis_idx]
        boundary_indices = np.where(ftet_vertices[:, axis_idx] <= 1e-3 + cutoff)[0]

    boundary_rest_pos = ftet_vertices[boundary_indices]
    # Fixed boundary: every frame holds the rest positions.
    boundary_pos_seq = np.stack([boundary_rest_pos] * (NUM_FRAMES + 1), axis=0)
    return _save(config_dir, seq_name, boundary_indices, boundary_rest_pos, boundary_pos_seq)


def save_pull_farthest_points(ftet_vertices, surface_vertices, config_dir,
                              num_points=4, offset_percentage=10):
    """Pull outward at `num_points` well-separated surface points and their neighborhoods."""
    seq_name = f"pull_farthest_points_{num_points}_{offset_percentage}percent"

    object_center = 0.5 * (surface_vertices.max(axis=0) + surface_vertices.min(axis=0))
    object_size = (surface_vertices.max(axis=0) - surface_vertices.min(axis=0)).max()
    boundary_point_indices = farthest_point_sampling(surface_vertices, num_points)

    kdtree = cKDTree(surface_vertices)
    pull_direction = []
    boundary_indices = []
    for boundary_index in boundary_point_indices:
        neighbors = kdtree.query_ball_point(surface_vertices[boundary_index], r=0.05 * object_size)
        boundary_indices.append(neighbors)
        for _ in neighbors:
            pull_direction.append((surface_vertices[boundary_index] - object_center)[None, :])

    boundary_indices = np.concatenate(boundary_indices, axis=0)
    pull_direction = np.concatenate(pull_direction, axis=0)
    pull_direction = pull_direction / np.linalg.norm(pull_direction, axis=1, keepdims=True)
    boundary_indices, unique_idx = np.unique(boundary_indices, return_index=True)
    pull_direction = pull_direction[unique_idx]

    boundary_rest_pos = surface_vertices[boundary_indices]
    # Re-index onto the volume mesh, which is what the solvers step.
    boundary_indices = cKDTree(ftet_vertices).query(boundary_rest_pos, k=1)[1]

    pull_distance = (offset_percentage / 100.0) * object_size
    ramp = np.linspace(0, 1, NUM_FRAMES + 1).reshape(-1, 1, 1)
    boundary_pos_seq = (boundary_rest_pos.reshape(1, -1, 3)
                        + pull_direction.reshape(1, -1, 3) * ramp * pull_distance)
    return _save(config_dir, seq_name, boundary_indices, boundary_rest_pos, boundary_pos_seq)


def save_pull_boundary(ftet_vertices, surface_vertices, config_dir,
                       axis="longest", percentage=5, offset_percentage=10):
    """Pull the two ends of an axis apart."""
    seq_name = f"pull_boundary_{axis}_{percentage}percent_offset_{offset_percentage}percent"

    if axis == "longest":
        axis_idx = int(np.argmax(surface_vertices.max(axis=0) - surface_vertices.min(axis=0)))
    elif axis in ("x", "y", "z"):
        axis_idx = "xyz".index(axis)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    ratio = percentage / 100.0
    sorted_indices = np.argsort(surface_vertices[:, axis_idx])
    num_vertices = surface_vertices.shape[0]
    min_cutoff = surface_vertices[sorted_indices[int(num_vertices * ratio)], axis_idx]
    max_cutoff = surface_vertices[sorted_indices[-int(num_vertices * ratio) + 1], axis_idx]
    min_indices = np.where(surface_vertices[:, axis_idx] <= 1e-3 + min_cutoff)[0]
    max_indices = np.where(surface_vertices[:, axis_idx] >= -1e-3 + max_cutoff)[0]

    boundary_indices = np.concatenate([min_indices, max_indices], axis=0)
    boundary_rest_pos = surface_vertices[boundary_indices]

    offset_direction_min = np.zeros(3)
    offset_direction_min[axis_idx] = -1.0
    offset_direction_max = np.zeros(3)
    offset_direction_max[axis_idx] = 1.0

    object_size = (surface_vertices.max(axis=0) - surface_vertices.min(axis=0)).max()
    ramp = np.linspace(0, 1, NUM_FRAMES + 1).reshape(-1, 1, 1)
    scale = (offset_percentage / 100.0) * object_size
    boundary_pos_seq = np.concatenate([
        surface_vertices[min_indices].reshape(1, -1, 3)
        + offset_direction_min.reshape(1, 1, 3) * ramp * scale,
        surface_vertices[max_indices].reshape(1, -1, 3)
        + offset_direction_max.reshape(1, 1, 3) * ramp * scale,
    ], axis=1)

    boundary_indices = cKDTree(ftet_vertices).query(boundary_rest_pos, k=1)[1]
    return _save(config_dir, seq_name, boundary_indices, boundary_rest_pos, boundary_pos_seq)


def _save(config_dir, seq_name, boundary_indices, boundary_rest_pos, boundary_pos_seq):
    os.makedirs(config_dir, exist_ok=True)
    out = os.path.join(config_dir, f"{seq_name}.npz")
    np.savez(out, seq_name=seq_name, num_frames=NUM_FRAMES,
             boundary_indices=boundary_indices,
             boundary_rest_pos=boundary_rest_pos,
             boundary_pos_seq=boundary_pos_seq)
    print(f"  BC: {seq_name} ({boundary_rest_pos.shape[0]} pts) -> {out}")
    return out
