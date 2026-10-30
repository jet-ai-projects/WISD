# Third-party notices

WISD builds on [Lightning Weave](https://github.com/gbcfchc/Lightning-Weave-dev), [Lightning OPD](https://github.com/jet-ai-projects/Lightning-OPD)
and [slime](https://github.com/THUDM/slime). The included `slime/` runtime,
`slime_plugins/`, checkpoint tools, and offline data processing contain modified
upstream source. Their Apache-2.0 license is included in [LICENSE](LICENSE).
Inherited NVIDIA copyright notices are retained in [NOTICE](NOTICE), rather
than repeated as per-file banners. This repository does not imply upstream
endorsement.

## Included code

| Source | Included files | Attribution |
| --- | --- | --- |
| Lightning OPD / slime | `slime/`, `slime_plugins/`, `train.py`, checkpoint tools | Apache-2.0; see NOTICE |
| [verl](https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py) | `slime/utils/seqlen_balancing.py` | Copyright 2024 Bytedance Ltd. and/or its affiliates; Apache-2.0 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py) | `slime/rollout/rm_hub/math_dapo_utils.py` | Copyright 2024 Bytedance Ltd. and/or its affiliates; Copyright 2022 EleutherAI and the HuggingFace Inc. team; Apache-2.0 |
| [slime Qwen3.5 support](https://github.com/THUDM/slime/tree/41014d1f29e201137fdffce737bb8bac65bc5219) | Qwen3.5 model plugins and conversion adapters | Upstream adaptation notes are retained in the corresponding files |

The AIME evaluation prompt/task configurations are adapted from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
distributed under the MIT License; see
[licenses/lm-evaluation-harness.txt](licenses/lm-evaluation-harness.txt).
The HMMT prompt configuration comes from the research evaluation setup.
The new evaluation adapters use the public math-verification and LiveCodeBench
APIs; no vendored LiveCodeBench or Pebble implementation is included.

## Changes in this distribution

This release adds weak-model construction by RTN quantization and magnitude
pruning, row-norm restoration, same-model scoring with full-vocabulary
normalization, and WISD training/evaluation launchers. It retains the shared
offline DOPD data format and tilted-target training runtime. Experiment-specific scheduling,
internal paths, run logs, and checkpoint artifacts are not distributed.

The numerical weak-model, scoring, and loss kernels are carried over from the
research implementation. Public-facing entrypoints and packaging have been
rewritten. The `offline_direct_opd` names remain in low-level modules to preserve
the serialized dataset format and configuration interfaces.

## External dependencies and data

PyTorch, Megatron-LM, mbridge, Megatron-Bridge, SGLang, vLLM, lm-evaluation-harness,
LiveCodeBench, and other dependencies are installed separately and retain their
own licenses. Model weights, training datasets, and benchmark data are not
included. Downloading and using each artifact remains subject to its own license
and access conditions.
