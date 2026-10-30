#!/usr/bin/env bash
# Shared paths for the offline data pipeline.

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON=${PYTHON:-python3}
TASK=${TASK:-math}
case "${TASK}" in
    math) DEFAULT_PROMPT_LENGTH=1024; DEFAULT_LABEL_KEY=reward_model ;;
    code) DEFAULT_PROMPT_LENGTH=4096; DEFAULT_LABEL_KEY=label ;;
    *) echo "TASK must be math or code" >&2; exit 2 ;;
esac
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/outputs/${TASK}}
SOURCE_DATASET=${SOURCE_DATASET:-${RUN_DIR}/prompts.parquet}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
NUM_PROMPTS=${NUM_PROMPTS:-3200}
RESPONSES_PER_PROMPT=${RESPONSES_PER_PROMPT:-4}
EXPECTED_ROWS=$((NUM_PROMPTS * RESPONSES_PER_PROMPT))
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-${DEFAULT_PROMPT_LENGTH}}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
DTYPE=${DTYPE:-bfloat16}
case "${DTYPE}" in
    bfloat16|float16) ;;
    *) echo "DTYPE must be bfloat16 or float16" >&2; exit 2 ;;
esac

configure_model() {
    : "${STUDENT_MODEL:?Set STUDENT_MODEL to a downloaded Hugging Face model}"
    STUDENT_REVISION=${STUDENT_REVISION:-${STUDENT_MODEL##*/}}
    WEAK_ANCHOR=${WEAK_ANCHOR:-rtn3_g128}
    case "${WEAK_ANCHOR}" in
        rtn3_g128|group_prune_g128_tau1of6_rownorm|group_prune_g128_tau1of6) ;;
        *) echo "Unknown WEAK_ANCHOR: ${WEAK_ANCHOR}" >&2; exit 2 ;;
    esac
    ANCHOR_DIR=${ANCHOR_DIR:-${RUN_DIR}/anchors/${WEAK_ANCHOR}}
    ASSET_LOCK=${ASSET_LOCK:-${RUN_DIR}/assets.json}
}

ensure_asset_lock() {
    if [[ ! -f "${ASSET_LOCK}" ]]; then
        "${PYTHON}" data_curation/prepare_direct_opd_assets.py \
            --student "${STUDENT_MODEL}" \
            --student-revision "${STUDENT_REVISION}" \
            --post-teacher "${STUDENT_MODEL}" \
            --post-teacher-revision "${STUDENT_REVISION}" \
            --pre-teacher "${STUDENT_MODEL}" \
            --pre-teacher-revision "${STUDENT_REVISION}" \
            --output "${ASSET_LOCK}"
    fi
}
