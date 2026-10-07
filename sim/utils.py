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

"""Shared utilities for simulation scripts."""

import logging
import numpy as np
import torch
import kaolin as kal

logger = logging.getLogger(__name__)


def sample_interior_uniform_grid(mesh, num_samples, device, min_interior_points=1000):
    """Sample interior points on a uniform grid inside a mesh.

    If the initial grid yields fewer than min_interior_points, the grid is
    refined automatically.

    Args:
        mesh: kaolin SurfaceMesh with .vertices and .faces on device.
        num_samples: Target number of grid points (cube root determines resolution).
        device: torch device.
        min_interior_points: Minimum acceptable interior points before refining.

    Returns:
        pts: Interior points, shape (N, 3).
        approx_vol: Approximate volume of the object.
    """
    verts = mesh.vertices
    box_size = (verts.max(dim=0).values - verts.min(dim=0).values).max()
    box_center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2

    nX = int(np.round(num_samples ** (1 / 3)))

    def _make_grid(nX):
        x = ((torch.arange(0, nX, device=device) + 0.5) / nX - 0.5) * box_size + box_center[0]
        y = ((torch.arange(0, nX, device=device) + 0.5) / nX - 0.5) * box_size + box_center[1]
        z = ((torch.arange(0, nX, device=device) + 0.5) / nX - 0.5) * box_size + box_center[2]
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    uniform_pts = _make_grid(nX)
    boolean_signs = kal.ops.mesh.check_sign(
        verts.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    n_interior = int(boolean_signs.sum().item())
    if n_interior < min_interior_points:
        nX_prev = nX
        if n_interior > 0:
            nX = int((num_samples * min_interior_points / n_interior) ** (1 / 3))
        else:
            raise ValueError("No interior points found in the mesh.")
        logger.warning(
            "Only %d interior grid points (<%d). Increasing nX from %d to %d and resampling.",
            n_interior, min_interior_points, nX_prev, nX)
        uniform_pts = _make_grid(nX)
        boolean_signs = kal.ops.mesh.check_sign(
            verts.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]
    approx_vol = (box_size ** 3) * boolean_signs.sum() / boolean_signs.numel()
    return pts, approx_vol


# Point-in-mesh sampling follows the usage pattern in Kaolin's Simplicits
# tutorial (examples/tutorial/physics/, Apache-2.0).
def sample_interior_random_volume(mesh, num_samples, device):
    """Sample random points inside a mesh volume.

    Args:
        mesh: kaolin SurfaceMesh with .vertices and .faces on device.
        num_samples: Target number of interior points.
        device: torch device.

    Returns:
        pts: Interior points, shape (N, 3). N may differ from num_samples.
        approx_vol: Approximate volume of the object.
    """
    verts = mesh.vertices
    bb_min = verts.min(dim=0).values
    bb_max = verts.max(dim=0).values
    uniform_pts = torch.rand(num_samples, 3, device=device) * (bb_max - bb_min) + bb_min
    boolean_signs = kal.ops.mesh.check_sign(
        verts.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)
    pts = uniform_pts[boolean_signs.squeeze()]
    bb_vol = (bb_max - bb_min).prod()
    approx_vol = bb_vol * boolean_signs.sum() / boolean_signs.numel()
    return pts, approx_vol
