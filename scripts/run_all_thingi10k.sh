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

# Run full pipeline for all 20 thingi10k examples.
# Usage: bash scripts/run_all_thingi10k.sh
set -e
cd "$(dirname "$0")/.."

tail -n +2 data/thingi10k_20examples.csv | while read FID YMS FIX_SIDE REST; do
    echo ""
    echo "============================================================"
    echo "  $FID (ym=$YMS, fix=$FIX_SIDE)"
    echo "============================================================"
    uv run python scripts/run_thingi10k_example.py --fid "$FID" --ym "$YMS" --fix-side "$FIX_SIDE"
done

echo ""
echo "============================================================"
echo "  Summarizing results"
echo "============================================================"
uv run python eval/summarize_errors.py --dataset thingi10k
