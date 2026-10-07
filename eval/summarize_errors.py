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
Summarize vertex error statistics across examples and methods.

Scans datasets/*/output/ for JSON error stats files and prints
a comparison table grouped by boundary condition type.

Usage:
    uv run python eval/summarize_errors.py [--dataset thingi10k]
"""

import argparse
import os
import json
import glob
import numpy as np
from collections import defaultdict


def get_bc_type(bc_name):
    """Classify boundary condition name into a type."""
    if 'fix_' in bc_name:
        return 'fix_side'
    elif 'pull_farthest' in bc_name:
        return 'pull_farthest'
    elif 'pull_boundary' in bc_name:
        return 'pull_boundary'
    elif 'twist' in bc_name:
        return 'twist'
    return 'other'


def discover_results(dataset_dir):
    """Find all vertex_error_stats.json files in a dataset output directory.

    Returns list of dicts with: file_id, method, bc_config, path, stats.
    """
    results = []
    output_dir = os.path.join(dataset_dir, "output")
    if not os.path.isdir(output_dir):
        return results

    for file_id in sorted(os.listdir(output_dir)):
        id_dir = os.path.join(output_dir, file_id)
        if not os.path.isdir(id_dir):
            continue
        for json_file in sorted(glob.glob(os.path.join(id_dir, "*_vertex_error_stats.json"))):
            basename = os.path.basename(json_file)
            # Pattern: sim_result_{method_model}_{bc_config}_vertex_error_stats.json
            prefix = "sim_result_"
            suffix = "_vertex_error_stats.json"
            if not basename.startswith(prefix) or not basename.endswith(suffix):
                continue
            middle = basename[len(prefix):-len(suffix)]
            # Split into method and bc_config
            # Patterns: rkpm_model_*, rkpm_h6_model_*, mlp_model_*, mlp_h16_model_*, etc.
            import re
            m = re.match(r'(rkpm(?:_h\d+)?_model)_(.*)', middle)
            if m:
                method = m.group(1).upper().replace("_MODEL", "").replace("_", " ")
                bc_config = m.group(2)
            else:
                m = re.match(r'(mlp(?:_h\d+)?_model)_(.*)', middle)
                if m:
                    method = m.group(1).upper().replace("_MODEL", "").replace("_", " ")
                    bc_config = m.group(2)
                else:
                    method = "unknown"
                    bc_config = middle

            with open(json_file) as f:
                stats = json.load(f)

            results.append({
                'file_id': file_id,
                'method': method,
                'bc_config': bc_config,
                'bc_type': get_bc_type(bc_config),
                'path': json_file,
                'mse': stats.get('normalized_mean_squared_error'),
                'rmse': stats.get('normalized_rmse'),
                'relative_mse': np.mean(stats.get('relative_frame_errors', [])) if stats.get('relative_frame_errors') else None,
                'max_frame_error': max(stats.get('normalized_frame_errors', [0])),
                'num_frames': stats.get('num_frames'),
            })
    return results


def print_comparison(results, methods=None):
    """Print side-by-side comparison table."""
    if methods is None:
        methods = sorted(set(r['method'] for r in results))

    # Group by (file_id, bc_config)
    grouped = defaultdict(dict)
    for r in results:
        key = (r['file_id'], r['bc_config'])
        grouped[key][r['method']] = r

    # Group by bc_type
    by_bc_type = defaultdict(list)
    for key in sorted(grouped.keys()):
        bc_type = grouped[key][next(iter(grouped[key]))]['bc_type']
        by_bc_type[bc_type].append(key)

    for bc_type in ['fix_side', 'pull_farthest', 'pull_boundary', 'twist', 'other']:
        keys = by_bc_type.get(bc_type, [])
        if not keys:
            continue

        print(f"\n{'─' * 100}")
        print(f"  {bc_type.upper().replace('_', ' ')}")
        print(f"{'─' * 100}")

        # Header
        method_cols = "  ".join(f"{m:>20s}" for m in methods)
        print(f"{'File ID':<12} {'BC Config':<35} {method_cols}")
        print(f"{'-'*12} {'-'*35} {'  '.join('-'*20 for _ in methods)}")

        method_mses = {m: [] for m in methods}

        for key in keys:
            file_id, bc_config = key
            vals = []
            for m in methods:
                r = grouped[key].get(m)
                if r and r['mse'] is not None:
                    vals.append(f"{r['mse']:.2e}")
                    method_mses[m].append(r['mse'])
                else:
                    vals.append("N/A")
            method_str = "  ".join(f"{v:>20s}" for v in vals)
            print(f"{file_id:<12} {bc_config:<35} {method_str}")

        # Summary
        print(f"{'-'*12} {'-'*35} {'  '.join('-'*20 for _ in methods)}")
        means = []
        for m in methods:
            if method_mses[m]:
                means.append(f"{np.mean(method_mses[m]):.2e}")
            else:
                means.append("N/A")
        mean_str = "  ".join(f"{v:>20s}" for v in means)
        print(f"{'Mean':<12} {'':<35} {mean_str}")

    # Overall summary
    print(f"\n{'═' * 100}")
    print("OVERALL SUMMARY (MSE / Max Frame Error)")
    print(f"{'═' * 100}")
    header = "  ".join(f"{m:>25s}" for m in methods)
    print(f"{'BC Type':<20} {header}")
    print(f"{'-'*20} {'  '.join('-'*25 for _ in methods)}")

    for bc_type in ['fix_side', 'pull_farthest', 'pull_boundary', 'twist']:
        keys = by_bc_type.get(bc_type, [])
        if not keys:
            continue
        vals = []
        for m in methods:
            mses = [grouped[k][m]['mse'] for k in keys if m in grouped[k] and grouped[k][m]['mse'] is not None]
            maxes = [grouped[k][m]['max_frame_error'] for k in keys if m in grouped[k] and grouped[k][m]['max_frame_error'] is not None]
            if mses:
                vals.append(f"{np.mean(mses):.2e} / {np.mean(maxes):.2e}")
            else:
                vals.append("N/A")
        val_str = "  ".join(f"{v:>25s}" for v in vals)
        print(f"{bc_type:<20} {val_str}")


def main():
    parser = argparse.ArgumentParser(description='Summarize vertex error statistics')
    parser.add_argument('--dataset', type=str, default='thingi10k',
                        choices=['thingi10k', 'simready', 'beam'],
                        help='Dataset to summarize')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_map = {
        'thingi10k': os.path.join(project_root, 'datasets', 'Thingi10K'),
        'simready': os.path.join(project_root, 'datasets', 'NVSimReady'),
        'beam': os.path.join(project_root, 'datasets', 'Beam'),
    }

    dataset_dir = dataset_map[args.dataset]
    print(f"Scanning {dataset_dir}/output/ ...")
    results = discover_results(dataset_dir)
    print(f"Found {len(results)} result files")

    if results:
        print_comparison(results)
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
