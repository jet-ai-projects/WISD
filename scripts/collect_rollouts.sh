#!/usr/bin/env bash
# Collect the frozen original model trajectories for self-distillation.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "STUDENT_MODEL=... bash scripts/collect_rollouts.sh"
    echo "Optional: TASK, RUN_DIR, NUM_PROMPTS, DTYPE, TP_SIZE, RANK, WORLD_SIZE."
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_pipeline.sh"
configure_model
ensure_asset_lock

extra_args=(--enable-thinking)
if [[ "${ENABLE_THINKING:-1}" == 0 ]]; then
    extra_args=(--no-enable-thinking)
fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == 1 ]]; then
    extra_args+=(--language-model-only)
fi
"${PYTHON}" data_curation/collect_direct_opd_rollouts.py \
    --model "${STUDENT_MODEL}" --model-revision "${STUDENT_REVISION}" \
    --asset-lock "${ASSET_LOCK}" --input "${SOURCE_DATASET}" \
    --output-dir "${ROLLOUT_DIR}" --label-key "${DEFAULT_LABEL_KEY}" \
    --max-prompts "${NUM_PROMPTS}" \
    --responses-per-prompt "${RESPONSES_PER_PROMPT}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --top-k 16 --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-1.0}" \
    --seed "${SEED:-42}" --batch-size "${COLLECT_BATCH_SIZE:-64}" \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --dtype "${DTYPE}" --max-num-seqs "${MAX_NUM_SEQS:-1024}" \
    --shard-size "${SHARD_SIZE:-1024}" \
    --rank "${RANK:-0}" --world-size "${WORLD_SIZE:-1}" \
    "${extra_args[@]}"
