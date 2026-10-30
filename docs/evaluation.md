# Evaluation

Evaluate the exported Hugging Face checkpoints. Use the mathematics student for
AIME24, AIME25, and HMMT25, and the code student for LiveCodeBench v5 and v6.

| Benchmark | Questions | Responses per question | Maximum output | Total context |
|---|---:|---:|---:|---:|
| AIME24 | 30 | 32 | 40,960 | 49,152 |
| AIME25 | 30 | 32 | 40,960 | 49,152 |
| HMMT25 | 30 | 32 | 40,960 | 49,152 |
| LiveCodeBench v5, from 2024-08-01 | 279 | 4 | 40,960 | 49,152 |
| LiveCodeBench v6, from 2025-02-01 | 131 | 4 | 40,960 | 49,152 |

Sampling uses temperature 0.6, top-p 0.95, and no top-k truncation. Accuracy is
correct responses divided by all sampled responses: **Avg@32** for mathematics
and **Avg@4** for code. It is not pass@32 or pass@4. Math Avg. weights the three
math benchmarks equally, Code Avg. weights the two code benchmarks equally, and
the overall average weights all five benchmarks equally.

These are the release defaults for new evaluations. The paper also includes
historical results with their recorded context/output budgets. Record the
checkpoint, prompt family, runtime versions, sampling seed, and budgets with each
new run, and use the same settings for its original-model baseline.

## Mathematics

Install the evaluation dependencies in a separate environment and a vLLM build
that supports the model architecture and GPU:

```bash
pip install -r requirements-eval.txt
MODEL=checkpoints/student-math-hf MATH_PROMPT_FAMILY=qwen3 TP_SIZE=1 \
  OUTPUT_DIR=results/math bash evaluation/eval_math.sh
```

`MATH_PROMPT_FAMILY` selects the prompt used in the paper:

| Value | AIME24 | AIME25 | HMMT25 |
|---|---|---|---|
| `qwen3` or `qwen36` | Question–answer | Question–answer | Step-by-step |
| `qwen35` or `olmo3` | Question–answer | Step-by-step | Step-by-step |

The short prompt is `Question: {problem}\nAnswer:`. The step-by-step prompt asks
for a final `Answer: \boxed{$Answer}` line. Full strings are in
`evaluation/math_tasks/`. The model's chat template wraps each user message.
The task YAMLs preserve all 32 responses through an identity filter, then use
`math_verify.parse` and `verify` to grade each full response.

Set `TASKS=wisd_aime24` to evaluate one task. For AIME25, the task names are
`wisd_aime25` and `wisd_aime25_step`. `MAX_MODEL_LEN` controls context size, and
`MODEL_ARGS` appends model-specific vLLM arguments. The output budget and repeat
count are declared in the task YAMLs.

Summarize the sample file emitted by lm-evaluation-harness:

```bash
python -m evaluation.summarize \
  --math-samples results/math/MODEL/samples_wisd_aime24_TIMESTAMP.jsonl \
  --output results/math/aime24-summary.json
```

Use the actual filename emitted by the harness. The summary reports accuracy,
problem count, response count, and repeats. Add `--tokenizer MODEL` to include
mean response length. GPU runtimes and request scheduling affect sampled outputs,
so the portable launcher is not a replay of every historical job.

## LiveCodeBench data and generation

Install the official [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
package in a compatible evaluation environment. Record the checkout revision.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git external/LiveCodeBench
pip install -e external/LiveCodeBench
git -C external/LiveCodeBench rev-parse HEAD

python -m evaluation.livecodebench_v6.lock_dataset \
  --release-version release_v6 --output data/evaluation/lcb-v6.lock.json
python -m evaluation.generate_code \
  --model checkpoints/student-code-hf \
  --dataset-lock data/evaluation/lcb-v6.lock.json \
  --chat-format codeqwen --output results/code/v6-generations.json
```

For v5, use `--release-version release_v5` and separate lock/output filenames.
`--revision` pins a Hugging Face dataset revision. `--raw-dir` accepts an existing
directory containing the release's `test.jsonl`, `test2.jsonl`, etc. Locks retain
source checksums and question IDs, and validate the expected problem count.

Both releases use the CodeQwenInstruct message content: the system message is
`You are a helpful assistant.`, followed by the problem and either a starter-code
or standard-input/output instruction. `evaluation/prompts.py` contains the exact
strings. `--chat-format codeqwen` uses the original Qwen3-4B wrapper. For OLMo and
the later Qwen architectures, use the model's native wrapper, for example:

```bash
python -m evaluation.generate_code \
  --model checkpoints/student-code-hf \
  --dataset-lock data/evaluation/lcb-v6.lock.json \
  --chat-format native --chat-template-kwargs '{"enable_thinking": true}' \
  --output results/code/v6-native-generations.json
```

Configure chat-template options supported by the selected tokenizer. Generation
supports `--tensor-parallel-size`, `--max-model-len`, `--max-new-tokens`,
`--repeats`, and `--seed`. `--llm-kwargs` supplies additional vLLM engine arguments.
It writes the model responses, extracted programs, generated-token counts, and a
companion protocol file recording the runtime and configuration. This step does
not execute the programs or deserialize private test bundles.

## LiveCodeBench scoring

The upstream checker executes generated Python. Run it in an isolated, disposable
container with networking disabled and only the evaluation files mounted. The
acknowledgement flag below does not create that isolation.

Use an image containing Python and the upstream LiveCodeBench package.
`RAW_DATA` is the directory of release JSONL files, `RESULT_DIR` contains the
saved generations, and `SCORING_IMAGE` names that image. All host paths must be
absolute. Recreate the lock inside the container so its raw-data paths resolve.

```bash
docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 32g --pids-limit 512 \
  --tmpfs /tmp:rw,nosuid,size=4g \
  --mount "type=bind,src=$(pwd)/evaluation,dst=/workspace/evaluation,readonly" \
  --mount "type=bind,src=${RAW_DATA},dst=/data,readonly" \
  --mount "type=bind,src=${RESULT_DIR},dst=/results" \
  --workdir /workspace --entrypoint bash "${SCORING_IMAGE}" -lc '
python -m evaluation.livecodebench_v6.lock_dataset \
  --release-version release_v6 --raw-dir /data \
  --output /results/lcb-v6-scoring.lock.json &&
python -m evaluation.score_code \
  --generation /results/v6-generations.json \
  --dataset-lock /results/lcb-v6-scoring.lock.json \
  --output /results/v6-evaluation.json \
  --workers 8 --acknowledge-isolated-execution'
```

The scorer uses the upstream test suite with a six-second timeout and reports
Avg@4 from individual candidate grades. Scoring runs on CPUs and does not load a
model. Keep the same upstream revision and raw dataset files for generation and
scoring.
