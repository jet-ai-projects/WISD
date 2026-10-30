#!/usr/bin/env bash
# Prepare math or code prompts. This does not generate training responses.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "TASK=math|code SOURCE_INPUT=/path/to/source bash scripts/prepare_data.sh"
    echo "Code also requires STUDENT_MODEL. Use LCB_LOCK or automatically download the benchmark lock."
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_pipeline.sh"
: "${SOURCE_INPUT:?Set SOURCE_INPUT to the math parquet or CodeSub-15K shard directory}"

if [[ "${TASK}" == math ]]; then
    "${PYTHON}" data_curation/prepare_direct_opd_skywork_math.py \
        --input "${SOURCE_INPUT}" --output "${SOURCE_DATASET}"
else
    : "${STUDENT_MODEL:?Code prompt selection requires a student tokenizer}"
    if [[ -z "${LCB_LOCK:-}" ]]; then
        LCB_LOCK=${RUN_DIR}/lcb-v6-lock.json
        lcb_source_args=()
        if [[ -n "${LCB_RAW_DIR:-}" ]]; then
            lcb_source_args=(--raw-dir "${LCB_RAW_DIR}")
        fi
        "${PYTHON}" evaluation/livecodebench_v6/lock_dataset.py \
            --release-version release_v6 --start-date 2025-02-01 \
            --expected-problems 131 --output "${LCB_LOCK}" \
            "${lcb_source_args[@]}"
    fi
    "${PYTHON}" data_curation/prepare_direct_opd_klear_code.py \
        --input-dir "${SOURCE_INPUT}" \
        --tokenizer "${STUDENT_MODEL}" --lcb-lock "${LCB_LOCK}" \
        --output "${SOURCE_DATASET}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-prompt-length "${MAX_PROMPT_LENGTH}" \
        --selection-seed "${SEED:-42}"
fi
