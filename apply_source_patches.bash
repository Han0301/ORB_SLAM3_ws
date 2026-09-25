#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
orbslam_dir="${workspace}/third_party/ORB_SLAM3"
patch_file="${workspace}/patches/orbslam3-gemini2l.patch"

git -C "${orbslam_dir}" apply --check "${patch_file}"
git -C "${orbslam_dir}" apply "${patch_file}"
echo "Applied Gemini 2L ORB-SLAM3 compatibility patch."
