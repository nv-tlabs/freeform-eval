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

# Download and extract Thingi10K ftetwild meshes into datasets/Thingi10K/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$(cd "$SCRIPT_DIR/../datasets" && pwd)/Thingi10K"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# Pre-generated tet meshes published by the fTetWild authors:
#   Hu, Schneider, Wang, Zorin, Panozzo. "Fast Tetrahedral Meshing in the Wild",
#   ACM TOG (SIGGRAPH 2020).  https://github.com/wildmeshing/fTetWild
# Their input is the Thingi10K dataset (https://ten-thousand-models.appspot.com/).
# Requires gdown: pip install gdown
if [ ! -f ftetwild_output_msh.tar.gz ]; then
    gdown 13zmGxikHiiSv9-eu8wZDTOWtPmR-KV5b -O ftetwild_output_msh.tar.gz
else
    echo "ftetwild_output_msh.tar.gz already exists, skipping download"
fi

# Extract (creates ftetwild_output_msh/ directory with .msh files)
tar xvzf ftetwild_output_msh.tar.gz
