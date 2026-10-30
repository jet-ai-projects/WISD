#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec "${PYTHON:-python3}" "${PROJECT_ROOT}/configs/wisd/train.py" "$@"
