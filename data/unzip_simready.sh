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

# Extract SimReady tet meshes and VoMP materials into datasets/NVSimReady/tet_npz/.
# The tar file is included in the repo via Git LFS (data/simready_tet_npz.tar.gz).
# After extracting, run data/export_simready.py to generate surface meshes.
#
# The original SimReady USD assets are available at:
#   https://d4i3qtqj3r0z5.cloudfront.net/Commercial_NVD%4010013.zip
#   https://d4i3qtqj3r0z5.cloudfront.net/Residential_NVD%4010012.zip
#   https://d4i3qtqj3r0z5.cloudfront.net/SimReady_Furniture_Misc_01_NVD%4010010.zip
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$(cd "$SCRIPT_DIR/../datasets" && pwd)/NVSimReady"
mkdir -p "$DEST_DIR"

tar xzf "$SCRIPT_DIR/simready_tet_npz.tar.gz" -C "$DEST_DIR"
echo "Extracted tet_npz to $DEST_DIR/tet_npz/"
