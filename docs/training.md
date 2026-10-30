# Training and checkpoint export

WISD trains the student on fixed original-model trajectories and cached
original-minus-weak scores. The student starts from the original checkpoint.
Only the student runs during optimization, and the trained student is used
alone at inference time.

The launcher uses the offline DOPD implementation in
`slime/rollout/offline_direct_opd.py`. It preserves the tilted target,
behavior-weighted log-probability loss, fixed Top16 support, and OTHER bucket
with zero reward. Runtime flags retain the `offline-direct-opd` naming to
match the existing cache format.

## Qwen3-4B reference recipe

Use a GPU environment with Slime, Megatron-LM, Transformer Engine, SGLang,
and Ray installed. Set `MEGATRON_PATH` when using a Megatron-LM source checkout.
The included `run_docker.sh` opens the Qwen3 training container with this
repository mounted at `/workspace/wisd-dev`.

```bash
export STUDENT_MODEL="$PWD/models/Qwen3-4B"
export MODEL_TYPE=qwen3-4B
export NUM_GPUS=8
export RUN_DIR="$PWD/outputs/qwen3-4b-math"
export WEAK_ANCHOR=rtn3_g128

OUTPUT_DIR="$PWD/checkpoints/qwen3-4b-initial" \
  bash scripts/convert_checkpoint.sh to-megatron

export DATA_DIR="$RUN_DIR/anchors/$WEAK_ANCHOR/final"
export LOAD_DIR="$PWD/checkpoints/qwen3-4b-initial"
export SAVE_DIR="$PWD/checkpoints/qwen3-4b-$WEAK_ANCHOR"

bash scripts/train.sh
```

Use the same `STUDENT_MODEL` checkpoint as in data preparation. The converter
creates the Megatron initialization from its original weights. `DATA_DIR`
contains the scored Parquet shards and `manifest.json` produced by
`scripts/score_anchors.sh`.

| Training parameter | Default |
|---|---|
| Distillation coefficient alpha | 2.0 |
| Learning rate / schedule | 1e-6 / constant |
| Optimizer / weight decay | Adam / 0.01 |
| Global batch size | 64 |
| Cached batch size | 256 trajectories |
| Cached batches / optimizer updates | 50 / 200 |
| Training precision | BF16 |
| Training seed | 1234 |

Each cached batch yields `256 / 64 = 4` optimizer updates. The default 50
batches consume the 12,800 trajectories once. `--optimizer-steps` specifies
an exact update budget, while `--num-rollout` specifies the number of cached
batches. `--save-interval` counts cached batches and defaults to 50.

For code training, set `TASK=code` and use the code cache. The launcher then
uses a 4,096-token prompt limit instead of the mathematics limit of 1,024.
Both use 2,048-token training responses. The model sequence limit covers
prompt plus response. If collection settings change, pass the matching
`--max-prompt-length`, `--max-response-length`, `--temperature`, `--top-p`,
`--top-k`, and `--responses-per-prompt` values to training.

## Command preview and Ray

```bash
bash scripts/train.sh --dry-run
```

The preview prints the resolved optimizer budget and commands without
starting Ray or loading models. A normal launch starts a local Ray head
bound to `127.0.0.1`, then submits the training job. Set `RAY_JOB_ADDRESS` to
use an existing dashboard address. The workers need access to the repository,
checkpoints, cache paths, and installed dependencies on the same machine.
The launcher does not upload the workspace to Ray. A local head remains
available after training and can be stopped with `ray stop` when no other
jobs use it.

The launcher accepts `--load` for initialization or an explicit training
resume. Keep the checkpoint, scored cache, and model identity consistent
when resuming. Use a new `SAVE_DIR` for a new student.

## Additional training adapters

The shared runtime also retains these model-specific training paths:

| `MODEL_TYPE` | Backend |
|---|---|
| `qwen3-4B` | Megatron |
| `qwen3.5-4B` | Megatron with the Qwen3.5 text-model adapter |
| `olmo3-7b-think` | Native Hugging Face model with FSDP |

Qwen3-4B is the integrated weak-construction and scoring example in this
release. The other entries expose the existing training adapters for compatible
cached data. Qwen3.5 requires its newer model/kernel stack. The recorded
adapter environment used Megatron Core 0.14, `fla-core==0.4.2`,
`flash-linear-attention==0.4.2`, and `transformers==5.5.3`. The converter uses
the `mbridge` module, which is distinct from `megatron.bridge`.

OLMo loads the original Hugging Face model directly, so skip the input
conversion and unset `LOAD_DIR` for a fresh run. Both backends use BF16
training with FP32 loss arithmetic.

## Export to Hugging Face

The default 50-batch run writes its final checkpoint as `iter_0000049`:

```bash
INPUT_DIR="$SAVE_DIR/iter_0000049" \
OUTPUT_DIR="$SAVE_DIR-hf" \
  bash scripts/convert_checkpoint.sh to-hf
```

The export combines the trained parameters with the original Hugging Face
configuration and tokenizer. Use the exported directory for evaluation.

## Local verification

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q
```

Tests cover numerical loss behavior, weak transforms, cache metadata, and
portable launcher arguments. The public launchers are extracted from the
research runtime and checked on CPU. A fresh GPU end-to-end run of the
packaged release is not included in those checks.
