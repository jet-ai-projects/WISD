#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to an HF model ID or checkpoint directory}"
OUTPUT_DIR=${OUTPUT_DIR:-results/math}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}
MATH_PROMPT_FAMILY=${MATH_PROMPT_FAMILY:-qwen3}
case "$MATH_PROMPT_FAMILY" in
    qwen3|qwen36) AIME25_TASK=wisd_aime25 ;;
    qwen35|olmo3) AIME25_TASK=wisd_aime25_step ;;
    *) echo "MATH_PROMPT_FAMILY must be qwen3, qwen35, olmo3, or qwen36" >&2; exit 2 ;;
esac
TASKS=${TASKS:-wisd_aime24,${AIME25_TASK},wisd_hmmt25}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Repetition is part of each task YAML, not a nonstandard --repeats CLI flag.
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TP_SIZE},max_model_len=${MAX_MODEL_LEN},dtype=bfloat16,gpu_memory_utilization=0.85,seed=42${MODEL_ARGS:+,${MODEL_ARGS}}" \
    --include_path "${ROOT}/evaluation/math_tasks" \
    --tasks "${TASKS}" \
    --batch_size auto \
    --apply_chat_template \
    --seed 42 \
    --log_samples \
    --output_path "${OUTPUT_DIR}"
