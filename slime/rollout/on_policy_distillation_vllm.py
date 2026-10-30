"""Online OPD reward adapter for a vLLM OpenAI-compatible teacher server."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from typing import Any

import aiohttp
import torch

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample


TEACHER_SERVED_MODEL = os.environ.get(
    "OPD_TEACHER_SERVED_MODEL",
    "olmo3-teacher",
)


@lru_cache(maxsize=None)
def _cached_tokenizer(path: str):
    return load_tokenizer(path, trust_remote_code=True)


def _teacher_scoring_tokens(args: Any, sample: Sample) -> list[int]:
    """Build a teacher-native prompt followed by an aligned response.

    When ``--rm-tokenizer-path`` is unset, retain the legacy same-tokenizer
    behavior.  Cross-template OPD must preserve the raw chat messages in the
    dataset and render them with the teacher tokenizer.  Response vocabulary
    symbols are mapped position by position because OPD consumes one teacher
    log-prob per student response token.
    """

    teacher_tokenizer_path = getattr(args, "rm_tokenizer_path", None)
    if not teacher_tokenizer_path:
        return [int(token_id) for token_id in sample.tokens]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    messages = metadata.get("_slime_prompt_messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "Cross-template online OPD requires raw prompt messages. "
            "Load the prompt dataset with --apply-chat-template so "
            "Dataset stores metadata['_slime_prompt_messages']."
        )
    if sample.response_length <= 0:
        raise ValueError("Cross-template online OPD requires a non-empty response")
    if len(sample.tokens) < sample.response_length:
        raise ValueError(
            "Student token sequence is shorter than response_length: "
            f"{len(sample.tokens)} < {sample.response_length}"
        )

    student_tokenizer = _cached_tokenizer(args.hf_checkpoint)
    teacher_tokenizer = _cached_tokenizer(teacher_tokenizer_path)
    student_response_ids = [
        int(token_id) for token_id in sample.tokens[-sample.response_length :]
    ]
    # Map the *generated token sequence*, rather than re-tokenizing
    # ``sample.response``.  SGLang keeps terminal special tokens (for example
    # EOS) in ``sample.tokens`` while omitting them from the response text.
    # Re-tokenizing that text therefore loses a supervised position.  Mapping
    # the tokenizer vocabulary symbols preserves both token boundaries and
    # terminal special tokens; cross-tokenizer OPD is rejected when such a
    # one-to-one mapping does not exist.
    student_response_tokens = student_tokenizer.convert_ids_to_tokens(
        student_response_ids
    )
    teacher_response_ids = teacher_tokenizer.convert_tokens_to_ids(
        student_response_tokens
    )
    if isinstance(teacher_response_ids, int):
        teacher_response_ids = [teacher_response_ids]
    if len(teacher_response_ids) != len(student_response_ids):
        raise ValueError(
            "Student/teacher response token length mismatch: "
            f"{len(student_response_ids)} != {len(teacher_response_ids)}"
        )
    teacher_response_ids = [int(token_id) for token_id in teacher_response_ids]
    recovered_teacher_tokens = teacher_tokenizer.convert_ids_to_tokens(
        teacher_response_ids
    )
    for position, (student_token, teacher_token) in enumerate(
        zip(student_response_tokens, recovered_teacher_tokens, strict=True)
    ):
        if student_token != teacher_token:
            raise ValueError(
                "Student/teacher response vocabulary-token mismatch at position "
                f"{position}: {student_token!r} != {teacher_token!r}"
            )

    tools = metadata.get("tools")
    teacher_prompt = teacher_tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    teacher_prompt_ids = teacher_tokenizer.encode(
        teacher_prompt,
        add_special_tokens=False,
    )
    return [
        *[int(token_id) for token_id in teacher_prompt_ids],
        *[int(token_id) for token_id in teacher_response_ids],
    ]


async def reward_func(args, sample: Sample, **kwargs):
    """Score the complete student sequence with vLLM prompt log-probabilities."""
    payload = {
        "model": TEACHER_SERVED_MODEL,
        "prompt": _teacher_scoring_tokens(args, sample),
        "echo": True,
        "max_tokens": 0,
        "logprobs": 1,
        "temperature": 0,
        "add_special_tokens": False,
    }
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as response:
            response.raise_for_status()
            return await response.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract response-token teacher log-probabilities from vLLM responses."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    scalar_rewards = [0.0] * len(samples)

    for sample, reward in zip(samples, raw_rewards, strict=True):
        try:
            token_logprobs = reward["choices"][0]["logprobs"]["token_logprobs"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"Malformed vLLM teacher response: {reward!r}"
            ) from error

        response_length = sample.response_length
        if len(token_logprobs) < response_length:
            raise ValueError(
                "Teacher prompt-logprob sequence is shorter than the student "
                f"response: {len(token_logprobs)} < {response_length}"
            )

        response_logprobs = token_logprobs[-response_length:]
        if any(value is None or not math.isfinite(float(value)) for value in response_logprobs):
            raise ValueError("Teacher returned a missing or non-finite response log-probability")

        sample.teacher_log_probs = torch.tensor(
            [float(value) for value in response_logprobs],
            dtype=torch.float32,
        )

    return scalar_rewards, scalar_rewards
