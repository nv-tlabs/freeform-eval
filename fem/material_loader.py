# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file was originally part of VoMP (https://github.com/nv-tlabs/VoMP)
# and has been modified. Changes include:
# - Replaced vomp.inference.utils.MaterialUpsampler with a local scipy cKDTree
#   implementation to remove the vomp runtime dependency
# - Added save_results and material_stats methods

import numpy as np
import warp as wp
import torch
# from vomp.inference.utils import MaterialUpsampler
from scipy.spatial import cKDTree
import os
from typing import Tuple, Dict, Any


def load_material_data(npz_path: str):
    """
    Load material data from npz file.
    
    Args:
        npz_path: Path to .npz file containing voxel_data
        
    Returns:
        tuple: (voxel_coords, voxel_materials) where:
            - voxel_coords: (N, 3) array of voxel positions
            - voxel_materials: (N, 3) array of [E, nu, rho] per voxel
    """
    data = np.load(npz_path)
    voxel_data = data['voxel_data']
    
    voxel_coords = np.stack([
        voxel_data['x'],
        voxel_data['y'],
        voxel_data['z']
    ], axis=1)
    
    voxel_materials = np.stack([
        voxel_data['youngs_modulus'],
        voxel_data['poissons_ratio'],
        voxel_data['density']
    ], axis=1)
    
    print(f"Loaded material data:")
    print(f"  Voxels: {voxel_coords.shape[0]}")
    print(f"  Young's modulus range: [{voxel_materials[:, 0].min():.2e}, {voxel_materials[:, 0].max():.2e}]")
    print(f"  Poisson's ratio range: [{voxel_materials[:, 1].min():.3f}, {voxel_materials[:, 1].max():.3f}]")
    print(f"  Density range: [{voxel_materials[:, 2].min():.2f}, {voxel_materials[:, 2].max():.2f}]")
    
    return voxel_coords, voxel_materials


def apply_spatially_varying_materials(sim, npz_path: str, k_neighbors: int = 1):
    voxel_coords, voxel_materials = load_material_data(npz_path)
    
    # Create upsampler
    upsampler = MaterialUpsampler(voxel_coords, voxel_materials)
    
    # Get vertex positions from the mesh
    node_positions = sim.lame_field.space.node_positions()
    query_points = node_positions.numpy()  # Convert warp array to numpy
    
    print(f"\nInterpolating materials to {query_points.shape[0]} mesh vertices...")
    
    interpolated_materials, distances = upsampler.interpolate(query_points, k=k_neighbors)
    
    youngs_modulus_per_vertex = interpolated_materials[:, 0]  # (N,) array
    poisson_ratio_per_vertex = interpolated_materials[:, 1]   # (N,) array
    density_per_vertex = interpolated_materials[:, 2]         # (N,) array
    
    # Convert Young's modulus and Poisson's ratio to Lame parameters
    # lame[0] = lambda = E * nu / ((1 + nu) * (1 - 2*nu))
    # lame[1] = mu = E / (2 * (1 + nu))
    lame_lambda = (youngs_modulus_per_vertex * poisson_ratio_per_vertex) / (
        (1.0 + poisson_ratio_per_vertex) * (1.0 - 2.0 * poisson_ratio_per_vertex)
    )
    lame_mu = youngs_modulus_per_vertex / (2.0 * (1.0 + poisson_ratio_per_vertex))
    
    # Stack into (N, 2) array for lame parameters [lambda, mu]
    lame_params = np.stack([lame_lambda, lame_mu], axis=1)
    
    # Directly set the lame field values (don't use scale_lame_field as it would multiply)
    # The lame_field.dof_values is a warp array of wp.vec2
    sim.lame_field.dof_values.assign(wp.array(lame_params, dtype=wp.vec2))
    sim.density_field.dof_values.assign(wp.array(density_per_vertex, dtype=sim.density_field.dof_values.dtype))
    
    return {
        'youngs_modulus': youngs_modulus_per_vertex,
        'poisson_ratio': poisson_ratio_per_vertex,
        'density': density_per_vertex,
        'interpolation_distances': distances
    }


def visualize_material_distribution(sim, material_stats: dict, output_path: str = None):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Young's modulus histogram
    axes[0].hist(material_stats['youngs_modulus'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("Young's Modulus (Pa)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Young's Modulus Distribution")
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Poisson's ratio histogram
    axes[1].hist(material_stats['poisson_ratio'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel("Poisson's Ratio")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Poisson's Ratio Distribution")
    axes[1].grid(True, alpha=0.3)
    
    # Density histogram
    axes[2].hist(material_stats['density'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2].set_xlabel("Density (kg/m³)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Density Distribution")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()

class MaterialUpsampler:
    """
    Nearest-neighbor material property interpolation for spatial upsampling.

    Efficiently interpolates material properties from sparse voxel data to arbitrary
    3D query points using scipy's cKDTree for fast spatial lookups.
    """

    def __init__(self, voxel_coords: np.ndarray, voxel_materials: np.ndarray):
        """
        Initialize the upsampler with voxel data.

        Args:
            voxel_coords: Voxel coordinates (N, 3) in world space
            voxel_materials: Material properties per voxel (N, 3) [E, nu, rho]
        """
        self.voxel_coords = np.asarray(voxel_coords, dtype=np.float32)
        self.voxel_materials = np.asarray(voxel_materials, dtype=np.float32)

        # Build KDTree for fast nearest neighbor search

        self.tree = cKDTree(self.voxel_coords)

    def interpolate(
        self, query_points: np.ndarray, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate material properties to query points via k-nearest neighbors.

        Args:
            query_points: Target coordinates (M, 3) for interpolation
            k: Number of neighbors for interpolation (1=NN, >1=weighted average)

        Returns:
            Tuple of (interpolated_materials, distances):
            - interpolated_materials: (M, 3) [E, nu, rho] at query points
            - distances: (M,) distance to nearest neighbor

        Raises:
            ValueError: If query_points shape is not (N, 3)
        """
        query_points = np.asarray(query_points, dtype=np.float32)

        if query_points.shape[1] != 3:
            raise ValueError(f"Query points must be (N, 3), got {query_points.shape}")

        # Find nearest neighbors
        distances, indices = self.tree.query(query_points, k=k)

        if k == 1:
            # Simple nearest neighbor
            interpolated_materials = self.voxel_materials[indices]
        else:
            # Distance-weighted average of k nearest neighbors
            weights = 1.0 / (
                distances + 1e-10
            )  # Add small epsilon to avoid division by zero
            weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights

            # Weighted average of materials
            interpolated_materials = np.sum(
                self.voxel_materials[indices] * weights[:, :, np.newaxis], axis=1
            )
            distances = distances[:, 0]  # Return distance to closest neighbor

        return interpolated_materials, distances

    def interpolate_to_gaussians(
        self, gaussian_model: "Gaussian"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to interpolate to Gaussian splat centers.

        Args:
            gaussian_model: Gaussian splat model

        Returns:
            Tuple of (interpolated_materials, distances)
        """
        # Get Gaussian positions
        gaussian_positions = gaussian_model.get_xyz.detach().cpu().numpy()

        return self.interpolate(gaussian_positions)

    def save_results(
        self,
        query_points: np.ndarray,
        materials: np.ndarray,
        distances: np.ndarray,
        output_path: str,
        format: str = "npz",
    ):
        """
        Save interpolation results to file.

        Args:
            query_points: Query coordinates (M, 3)
            materials: Interpolated materials (M, 3)
            distances: Distances to nearest neighbors (M,)
            output_path: Output file path
            format: File format ("npz" or "pth")
        """

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        if format.lower() == "npz":
            # Create structured array
            num_points = len(query_points)
            dtype = [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("youngs_modulus", "<f4"),
                ("poissons_ratio", "<f4"),
                ("density", "<f4"),
                ("nn_distance", "<f4"),
                ("segment_id", "<U32"),
            ]

            data = np.zeros(num_points, dtype=dtype)
            data["x"] = query_points[:, 0].astype(np.float32)
            data["y"] = query_points[:, 1].astype(np.float32)
            data["z"] = query_points[:, 2].astype(np.float32)
            data["youngs_modulus"] = materials[:, 0].astype(np.float32)
            data["poissons_ratio"] = materials[:, 1].astype(np.float32)
            data["density"] = materials[:, 2].astype(np.float32)
            data["nn_distance"] = distances.astype(np.float32)
            data["segment_id"] = "interpolated_material"

            np.savez_compressed(output_path, voxel_data=data)

        elif format.lower() == "pth":
            # Save as PyTorch tensors
            data = {
                "x": torch.from_numpy(query_points[:, 0]),
                "y": torch.from_numpy(query_points[:, 1]),
                "z": torch.from_numpy(query_points[:, 2]),
                "youngs_modulus": torch.from_numpy(materials[:, 0]),
                "poissons_ratio": torch.from_numpy(materials[:, 1]),
                "density": torch.from_numpy(materials[:, 2]),
                "nn_distance": torch.from_numpy(distances),
                "query_points": torch.from_numpy(query_points),
                "materials": torch.from_numpy(materials),
            }
            torch.save(data, output_path)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'pth'")

    @property
    def num_voxels(self) -> int:
        """Get number of voxels"""
        return len(self.voxel_coords)

    @property
    def material_stats(self) -> Dict[str, Any]:
        """Get material property statistics"""
        return {
            "youngs_modulus": {
                "mean": self.voxel_materials[:, 0].mean(),
                "std": self.voxel_materials[:, 0].std(),
            },
            "poisson_ratio": {
                "mean": self.voxel_materials[:, 1].mean(),
                "std": self.voxel_materials[:, 1].std(),
            },
            "density": {
                "mean": self.voxel_materials[:, 2].mean(),
                "std": self.voxel_materials[:, 2].std(),
            },
        }