#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ $# -ne 1 || "$1" == "--help" ]]; then
    echo "Usage: STUDENT_MODEL=... OUTPUT_DIR=... bash scripts/convert_checkpoint.sh to-megatron"
    echo "       STUDENT_MODEL=... INPUT_DIR=... OUTPUT_DIR=... bash scripts/convert_checkpoint.sh to-hf"
    echo "Options: MODEL_TYPE=qwen3-4B, NUM_GPUS=8, MEGATRON_PATH=..., DRY_RUN=1"
    exit 0
fi

: "${STUDENT_MODEL:?Set STUDENT_MODEL to the local Hugging Face model directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
MODEL_TYPE=${MODEL_TYPE:-qwen3-4B}
NUM_GPUS=${NUM_GPUS:-8}
MODEL_CONFIG=${MODEL_TYPE}
case "${MODEL_TYPE}" in
    qwen3-4B|qwen3.5-4B) ;;
    olmo3-7b-think) ;;
    *) echo "Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2; exit 2 ;;
esac

export PYTHONPATH="${PROJECT_ROOT}${MEGATRON_PATH:+:${MEGATRON_PATH}}${PYTHONPATH:+:${PYTHONPATH}}"
case "$1" in
    to-megatron)
        if [[ "${MODEL_TYPE}" == "olmo3-7b-think" ]]; then
            echo "OLMo uses the original Hugging Face model through FSDP; no input conversion is needed." >&2
            exit 2
        fi
        if [[ -e "${OUTPUT_DIR}" ]]; then
            echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
            exit 2
        fi
        source "${PROJECT_ROOT}/configs/models/${MODEL_CONFIG}.sh"
        COMMAND=(
            torchrun --nproc-per-node "${NUM_GPUS}"
            --master-port "${MASTER_PORT:-29500}"
            "${PROJECT_ROOT}/tools/convert_hf_to_torch_dist.py"
            "${MODEL_ARGS[@]}"
            --hf-checkpoint "${STUDENT_MODEL}" --save "${OUTPUT_DIR}"
        )
        ;;
    to-hf)
        : "${INPUT_DIR:?Set INPUT_DIR to the saved iter_XXXXXXX checkpoint directory}"
        CONVERTER=convert_torch_dist_to_hf.py
        EXTRA_ARGS=()
        if [[ "${MODEL_TYPE}" == "olmo3-7b-think" ]]; then
            CONVERTER=convert_fsdp_to_hf.py
            EXTRA_ARGS+=(--strict)
        fi
        COMMAND=(
            "${PYTHON:-python3}" "${PROJECT_ROOT}/tools/${CONVERTER}"
            --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
            --origin-hf-dir "${STUDENT_MODEL}" "${EXTRA_ARGS[@]}"
        )
        ;;
    *) echo "Expected to-megatron or to-hf, got: $1" >&2; exit 2 ;;
esac
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    exec "${COMMAND[@]}"
fi
