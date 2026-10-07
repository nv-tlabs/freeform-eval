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
Create skinning weights for a mesh (RKPM or MLP).

Usage:
    uv run python sim/create_model.py --config config/thingi10k/80597/create_rkpm.yaml
"""

import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import time

import numpy as np
import torch
import trimesh
import kaolin as kal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.utils import sample_interior_uniform_grid, sample_interior_random_volume
from kaolin.physics.simplicits import SimplicitsObject
from kaolin.physics.simplicits.training import PhysicsPoints
from kaolin.physics.simplicits.rkpm import SimplicitsRKPM
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('highest')

# Time the RKPM init (FPS node selection + eigenanalysis) without modifying kaolin source.
# This corresponds to "Training time" in Table 2 and "Time" in Table 3 of the paper.
_orig_rkpm_init = SimplicitsRKPM.init
def _timed_rkpm_init(self, pts, yms, prs, rhos, appx_vol):
    t0 = time.time()
    _orig_rkpm_init(self, pts, yms, prs, rhos, appx_vol)
    print(f"RKPM init time: {time.time() - t0:.2f}s")
SimplicitsRKPM.init = _timed_rkpm_init


def main():
    parser = argparse.ArgumentParser(description='Create Skinning Weights')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config values, e.g. num_handles=7 output_name=foo.pth')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda"
    dtype = torch.float32

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    method = cfg.get("method", "rkpm")

    # Load mesh
    mesh_path = os.path.join(project_root, cfg.surface_mesh)
    tri_mesh = trimesh.load(mesh_path, process=False)
    mesh = kal.rep.SurfaceMesh(
        vertices=torch.tensor(tri_mesh.vertices, device=device, dtype=dtype),
        faces=torch.tensor(tri_mesh.faces, device=device, dtype=torch.long),
    )

    # Sample interior points
    num_samples = cfg.get("num_sample_pts", 64 ** 3)
    sample_method = cfg.get("sample_method", "uniform_grid")
    if sample_method == "uniform_grid":
        so_pts, approx_volume = sample_interior_uniform_grid(mesh, num_samples, device)
    elif sample_method == "random_volume":
        so_pts, approx_volume = sample_interior_random_volume(mesh, num_samples, device)
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}")
    print(f"Sampled {so_pts.shape[0]} interior points, approx volume: {approx_volume.item():.4f}")

    # Material properties
    materials_path = cfg.get("materials", None)
    if materials_path is not None:
        # Spatially varying materials from VoMP npz
        materials_path = os.path.join(project_root, materials_path)
        mat_data = np.load(materials_path)
        voxel_data = mat_data["voxel_data"]
        mat_pts = torch.tensor(np.stack([voxel_data["x"], voxel_data["y"], voxel_data["z"]], axis=1),
                               device=device, dtype=dtype)
        mat_yms = torch.tensor(voxel_data["youngs_modulus"], device=device, dtype=dtype)
        mat_prs = torch.tensor(voxel_data["poissons_ratio"], device=device, dtype=dtype)
        mat_rhos = torch.tensor(voxel_data["density"], device=device, dtype=dtype)
        # Nearest-neighbor lookup
        from scipy.spatial import cKDTree
        tree = cKDTree(mat_pts.cpu().numpy())
        _, nn_idx = tree.query(so_pts.cpu().numpy(), k=1)
        so_yms = mat_yms[nn_idx]
        so_prs = mat_prs[nn_idx]
        so_rhos = mat_rhos[nn_idx]
        print(f"Spatially varying materials: YM [{so_yms.min():.2e}, {so_yms.max():.2e}], "
              f"PR [{so_prs.min():.3f}, {so_prs.max():.3f}], rho [{so_rhos.min():.1f}, {so_rhos.max():.1f}]")
    else:
        # Uniform materials
        ym = float(cfg.young_modulus)
        pr = float(cfg.poisson_ratio)
        rho = float(cfg.density)
        so_yms = torch.full((so_pts.shape[0],), ym, device=device, dtype=dtype)
        so_prs = torch.full((so_pts.shape[0],), pr, device=device, dtype=dtype)
        so_rhos = torch.full((so_pts.shape[0],), rho, device=device, dtype=dtype)
    physics_pts = PhysicsPoints(pts=so_pts, yms=so_yms, prs=so_prs, rhos=so_rhos, appx_vol=approx_volume)

    # Create model
    num_handles = int(cfg.num_handles)
    start_time = time.time()

    if method == "rkpm":
        num_nodes = int(cfg.num_nodes)
        print(f"Creating RKPM: {num_handles} handles, {num_nodes} nodes")
        simplicits_obj = SimplicitsObject.create_with_rkpm(
            physics_points=physics_pts,
            num_handles=num_handles,
            num_nodes=num_nodes,
            dtype=torch.float64,
        )
    elif method == "mlp":
        num_layers = int(cfg.num_layers)
        num_steps = int(cfg.num_steps)
        print(f"Training MLP: {num_handles} handles, {num_layers} layers, {num_steps} steps")
        simplicits_obj = SimplicitsObject.create_with_mlp(
            physics_points=physics_pts,
            num_handles=num_handles,
            num_samples=int(cfg.get("num_training_samples", 1000)),
            model_layers=num_layers,
            training_num_steps=num_steps,
            training_lr_start=float(cfg.get("lr_start", 1e-3)),
            training_lr_end=float(cfg.get("lr_end", 1e-3)),
            training_le_coeff=float(cfg.get("le_coeff", 1e-1)),
            training_lo_coeff=float(cfg.get("lo_coeff", 1e6)),
            training_log_every=1000,
            normalize_for_training=cfg.get("normalize_for_training", False),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    total_time = time.time() - start_time
    print(f"{method.upper()} creation time: {total_time:.2f}s")

    model = simplicits_obj.skinning_mod
    model.eval()
    with torch.no_grad():
        so_weights = model(so_pts)
    print(f"Weight matrix shape: {so_weights.shape}")
    print(f"w.T @ w cond: {torch.linalg.cond(so_weights.T @ so_weights).item():.4f}")

    # Save
    output_dir = os.path.join(project_root, cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_name = cfg.get("output_name", f"{method}_model.pth")
    output_path = os.path.join(output_dir, output_name)

    save_dict = {
        "so_pts": so_pts.cpu(),
        "so_yms": so_yms.cpu(),
        "so_prs": so_prs.cpu(),
        "so_rhos": so_rhos.cpu(),
        "so_appx_vol": approx_volume,
        "model": model,
        "weights": so_weights.cpu(),
        "num_handles": num_handles,
        "input_mesh_file": cfg.surface_mesh,
        "total_time": total_time,
        "method": method,
        "materials": cfg.get("materials", None),
    }
    torch.save(save_dict, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
