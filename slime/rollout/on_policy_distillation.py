import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    # For Lightning OPD: teacher log-probs are pre-computed in metadata,
    # no teacher server call needed. Return a sentinel so post_process_rewards knows.
    metadata = sample.metadata or {}
    if metadata.get("is_offline_direct_opd", False):
        return {"offline_direct_opd": True}
    if metadata.get("is_lightning_opd", False) or metadata.get("is_offline_opd", False):
        return {"lightning_opd": True}

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    For Lightning OPD, teacher log-probs are pre-computed in the parquet
    metadata instead of being fetched from a teacher server at runtime.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    for i, (sample, reward) in enumerate(zip(samples, raw_rewards)):
        metadata = sample.metadata or {}
        if isinstance(reward, dict) and reward.get("offline_direct_opd"):
            # Direct-OPD candidate scores are attached by the fixed-rollout loader
            # and consumed by its custom loss; no 1D OPD score is needed here.
            continue
        elif isinstance(reward, dict) and reward.get("lightning_opd"):
            # Lightning OPD: teacher log-probs are pre-computed in metadata
            pre_teacher_lp = metadata.get("teacher_log_probs", [])
            sample.teacher_log_probs = torch.tensor(
                [float(x) for x in pre_teacher_lp], dtype=torch.float32
            )
        else:
            # Online OPD: extract teacher log-probs from sglang response
            t_log_probs = torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            sample.teacher_log_probs = t_log_probs[-response_lengths[i]:]

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
