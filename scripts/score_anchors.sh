#!/usr/bin/env bash
# Score the original and weakened model on fixed trajectories, then seal the cache.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "STUDENT_MODEL=... WEAK_ANCHOR=rtn3_g128 bash scripts/score_anchors.sh"
    echo "Pruning: WEAK_ANCHOR=group_prune_g128_tau1of6_rownorm"
    echo "Original, weak, and reference scoring run sequentially on one GPU."
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_pipeline.sh"
configure_model
ensure_asset_lock

score() {
    local field=$1 input_dir=$2 output_dir=$3
    shift 3
    "${PYTHON}" data_curation/precompute_direct_opd_scores.py \
        --model "${STUDENT_MODEL}" --model-revision "${STUDENT_REVISION}" \
        --asset-lock "${ASSET_LOCK}" --score-field "${field}" \
        --input "${input_dir}" --output-dir "${output_dir}" \
        --dtype "${DTYPE}" --attn-implementation sdpa \
        --device "${SCORE_DEVICE:-cuda:0}" \
        --row-batch-size "${SCORE_BATCH_SIZE:-8}" \
        --chunk-size "${SCORE_CHUNK_SIZE:-4096}" \
        --rank 0 --world-size 1 "$@"
}

score post_teacher_log_probs \
    "${ROLLOUT_DIR}" "${ANCHOR_DIR}/post"
score pre_teacher_log_probs \
    "${ANCHOR_DIR}/post" "${ANCHOR_DIR}/pre" --weak-anchor "${WEAK_ANCHOR}"
score student_ref_sampled_log_probs \
    "${ANCHOR_DIR}/pre" "${ANCHOR_DIR}/final"

"${PYTHON}" data_curation/prepare_direct_opd_manifest.py \
    --input "${ANCHOR_DIR}/final" --source-dataset "${SOURCE_DATASET}" \
    --asset-lock "${ASSET_LOCK}" \
    --manifest-out "${ANCHOR_DIR}/final/manifest.json"
