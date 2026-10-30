#!/usr/bin/env bash
# Start an interactive, repository-local GPU workspace.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE=${IMAGE:-tonyhao96/jetmoe:v0.2}
exec docker run --rm -it --gpus all \
    --shm-size=64g \
    --mount "type=bind,source=${REPO_ROOT},target=/workspace/wisd-dev" \
    --workdir /workspace/wisd-dev \
    "${IMAGE}" bash
