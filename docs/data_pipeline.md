# Data preparation and weak-model scoring

The WISD pipeline is **prepare prompts → collect original-model trajectories →
score the original and weakened model → seal the cache → train the student**.
It uses a single original checkpoint. Quantization or pruning is applied only
to the frozen weak scorer, while the student starts from the original weights.

The integrated example uses Qwen3-4B. Run these commands from the repository
root in the curation environment.

## Model and output paths

```bash
export STUDENT_MODEL="$(hf download Qwen/Qwen3-4B)"
export TASK=math
export RUN_DIR="$PWD/outputs/qwen3-4b-math"
export WEAK_ANCHOR=rtn3_g128
```

`STUDENT_MODEL` is a local Hugging Face checkpoint directory. The wrappers
default `STUDENT_REVISION` to its basename, which is a commit SHA for standard
Hub snapshots. Set it explicitly when using another checkpoint layout.
The same checkpoint and revision populate the student, original-scoring, and
weak-scoring roles in `assets.json`.

## Mathematics prompts

```bash
hf download Skywork/Skywork-OR1-RL-Data \
  data/math-00000-of-00001.parquet \
  --repo-type dataset --local-dir data/skywork-or1

SOURCE_INPUT="$PWD/data/skywork-or1/data/math-00000-of-00001.parquet" \
  bash scripts/prepare_data.sh
```

The converter wraps each question in the mathematics instruction. Collection
selects the first `NUM_PROMPTS` questions that meet the prompt-length limit.
The default is 3,200 prompts, with four sampled responses each. Responses are
retained without correctness filtering.

## Code prompts

Mathematics and code use separate students and caches. For a code run:

```bash
export TASK=code
export RUN_DIR="$PWD/outputs/qwen3-4b-code"
hf download Kwai-Klear/KlearReasoner-CodeSub-15K \
  --repo-type dataset \
  --revision f6c935197e097ab144d5517a62cc498e0209e060 \
  --include 'default/train/*.parquet' --local-dir data/codesub15k

SOURCE_INPUT="$PWD/data/codesub15k/default/train" \
  bash scripts/prepare_data.sh
```

The code preparer filters by tokenizer length, removes exact and high-overlap
LiveCodeBench prompt matches, deduplicates normalized prompts, and selects
3,200 prompts after shuffling with seed 42. It keeps a hash of each source
label rather than executable test bundles. Ground-truth tests are not used
to train the student.

The wrapper obtains the LiveCodeBench v6 question list for decontamination.
Use `LCB_RAW_DIR` for a local raw-data directory or `LCB_LOCK` for a previously
prepared benchmark configuration.

## Collect the original model's trajectories

```bash
bash scripts/collect_rollouts.sh
```

| Setting | Default |
|---|---|
| Prompts / responses per prompt | 3,200 / 4 |
| Prompt limit, mathematics / code | 1,024 / 4,096 tokens |
| Response limit | 2,048 tokens |
| Sampling temperature / top-p | 1.0 / 1.0 |
| Sampling seed | 42 |
| Cached candidates per response position | 16 |
| Model-forward dtype | BF16 |

These are training-trajectory settings. Evaluation uses the separate budgets
in [evaluation.md](evaluation.md).

The cache contains response tokens, loss masks, fixed candidate IDs,
behavior log-probabilities, and sampled-token probabilities. All subsequent
scorers read these same prefixes and candidate IDs. `TP_SIZE` sets collection
tensor parallelism. Independent collectors can use distinct `RANK` values
with one shared `WORLD_SIZE`, and scoring starts after collection completes.

## Score a weak-model contrast

Quantization uses 3-bit symmetric round-to-nearest weights in groups of 128:

```bash
WEAK_ANCHOR=rtn3_g128 bash scripts/score_anchors.sh
```

Pruning removes weights at or below each group's maximum magnitude divided by six,
then restores the complete output row's original L2 norm:

```bash
WEAK_ANCHOR=group_prune_g128_tau1of6_rownorm bash scripts/score_anchors.sh
```

Both commands reuse the collected trajectories. They write separate outputs
under `anchors/<weak-anchor>/`, so both variants can share one rollout cache.
The optional `group_prune_g128_tau1of6` variant omits row-norm restoration.

Each command runs three sequential scoring passes:

1. The original model supplies `post_teacher_log_probs`.
2. A fresh copy of that model is weakened in memory and supplies
   `pre_teacher_log_probs` on the same candidates.
3. The original model supplies `student_ref_sampled_log_probs`.

The scoring passes use the same BF16/FP16 forward dtype and SDPA attention
backend. Probabilities are normalized over the full model vocabulary before
gathering the cached candidates. Transform arithmetic and probability
normalization use FP32. The original checkpoint files are unchanged.

The wrapper uses one GPU per scoring pass. `SCORE_DEVICE`,
`SCORE_BATCH_SIZE`, and `SCORE_CHUNK_SIZE` control device placement and memory
use. `DTYPE=float16` selects FP16 for both collection and scoring.

## Training hand-off

The final scoring pass writes Parquet shards and a manifest under:

```text
outputs/qwen3-4b-math/
  assets.json
  prompts.parquet
  rollouts/
  anchors/
    rtn3_g128/
      post/
      pre/
      final/
        manifest.json
        part-*.parquet
```

Pass the `final/` directory directly to training:

```bash
export DATA_DIR="$RUN_DIR/anchors/$WEAK_ANCHOR/final"
```

The reward is the original-minus-weak log-probability contrast. Training
retains the Top16 support and OTHER bucket, with zero reward assigned to
OTHER. There is no additional target-composition or trajectory-replay step.
See [training.md](training.md) for checkpoint conversion and the 200-update
recipe. Use fresh output directories when collecting or scoring a new run.
