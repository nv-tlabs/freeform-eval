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

# Run full pipeline for all 20 simready examples.
# Usage: bash scripts/run_all_simready.sh
set -e
cd "$(dirname "$0")/.."

tail -n +2 data/simready_20examples.csv | while read FID REST; do
    echo ""
    echo "============================================================"
    echo "  $FID"
    echo "============================================================"
    uv run python scripts/run_simready_example.py --fid "$FID"
done

echo ""
echo "============================================================"
echo "  Summarizing results"
echo "============================================================"
uv run python eval/summarize_errors.py --dataset simready
