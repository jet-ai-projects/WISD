"""LiveCodeBench message content used by WISD evaluations.

Formatting follows the upstream CodeQwenInstruct prompt. Model-specific chat
delimiters are applied separately by the generation entry point.
"""

SYSTEM_MESSAGE = "You are a helpful assistant."
CODE_PREFIX = (
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all "
    "tests. You will NOT return anything except for the program."
)
WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def code_messages(row):
    question = row["question_content"]
    starter = row.get("starter_code") or ""
    instruction = WITH_STARTER_CODE if starter else WITHOUT_STARTER_CODE
    code = starter if starter else "# YOUR CODE HERE"
    user = f"{CODE_PREFIX}\n\nQuestion: {question}\n\n{instruction}\n```python\n{code}\n```\n\n"
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]


def render_code_prompt(row, tokenizer, chat_format="codeqwen", chat_template_kwargs=None):
    messages = code_messages(row)
    if chat_format == "codeqwen":
        # This is the legacy LCB CodeQwenInstruct wrapper used for Qwen3-4B.
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if chat_format != "native":
        raise ValueError("chat_format must be codeqwen or native")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
