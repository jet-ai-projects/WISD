import copy
import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import ray

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset", "create_dataset"]

logger = logging.getLogger(__name__)


def _read_single_file(path, row_slice=None):
    """Read a single data file (jsonl or parquet)."""
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype={"label": str})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info(f"read_file path={path} slice {len(df)=} rows into {row_slice=}")
        df = df.iloc[row_slice]

    for _, row in df.iterrows():
        yield row.to_dict()


def _list_data_files(directory):
    """List all supported data files in a directory (recursively)."""
    supported_extensions = ('.jsonl', '.parquet')
    data_files = []

    for root, _, files in os.walk(directory):
        for file in sorted(files):  # Sort for deterministic order
            if file.endswith(supported_extensions):
                data_files.append(os.path.join(root, file))

    return sorted(data_files)  # Sort by full path for deterministic order


def read_file(path):
    """Read data from a file or directory.

    Args:
        path: Path to a data file (.jsonl or .parquet) or a directory containing data files.
            If a directory is provided, all .jsonl and .parquet files in it (and subdirectories)
            will be read and concatenated.
            Supports row slicing with @[start:end] suffix, e.g., "data.jsonl@[0:1000]"

    Yields:
        dict: Each row of data as a dictionary.
    """
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    # Handle directory: read all data files inside
    if os.path.isdir(path):
        data_files = _list_data_files(path)
        if not data_files:
            raise ValueError(f"No .jsonl or .parquet files found in directory: {path}")

        logger.info(f"Found {len(data_files)} data files in directory {path}")

        # For directory, row_slice applies to the combined dataset
        if row_slice is not None:
            # Collect all data first, then apply slice
            all_rows = []
            for file_path in data_files:
                for row in _read_single_file(file_path):
                    all_rows.append(row)
            logger.info(f"read_file directory={path} slice {len(all_rows)=} rows into {row_slice=}")
            for row in all_rows[row_slice]:
                yield row
        else:
            # Stream data from each file
            for file_path in data_files:
                for row in _read_single_file(file_path):
                    yield row
    else:
        # Handle single file
        for row in _read_single_file(path, row_slice):
            yield row


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(formatted_prompt: str, tokenizer, processor, max_length, multimodal_inputs=None):
    if max_length is None:
        return False

    if processor:
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        input_ids = processor_output["input_ids"][0]
    else:
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        assert as_conversation, "as_conversation must be True when multimodal_keys is not None"
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return prompt





class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        self.origin_samples = []

        for data in read_file(path):
            metadata = data.get(metadata_key) or {}

            prompt = _build_messages(data, prompt_key, apply_chat_template, multimodal_keys)

            # Cross-template reward models (for example an Instruct student
            # distilled from a Think teacher) must format the same source
            # messages with the teacher's own chat template.  Keep the
            # unformatted messages alongside the student-formatted prompt so
            # an online reward adapter can reconstruct that teacher prompt
            # without trying to parse rendered template text.
            if apply_chat_template and isinstance(prompt, list):
                metadata["_slime_prompt_messages"] = copy.deepcopy(prompt)

            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                formatted_prompt = prompt

            if processor:
                # temporary solution, will write image utils for slime later
                from qwen_vl_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                images, videos = process_vision_info(prompt)
                multimodal_inputs = {"images": images, "videos": videos}
            else:
                multimodal_inputs = None

            # TODO: this is slow.
            if _should_skip_prompt(formatted_prompt, tokenizer, processor, max_length, multimodal_inputs):
                continue

            self.origin_samples.append(
                Sample(
                    prompt=formatted_prompt,
                    label=data.get(label_key, None) if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        logger.info(f"Dataset: Loaded {len(self.origin_samples)} samples from {path}")
        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data



def create_dataset(
    paths,
    tokenizer,
    processor,
    max_length,
    *,
    prompt_key="text",
    multimodal_keys=None,
    label_key=None,
    tool_key=None,
    metadata_key="metadata",
    seed=42,
    apply_chat_template=False,
    apply_chat_template_kwargs=None,
):
    """Factory function to create a Dataset.

    Args:
        paths: A single path string, or a list with one path from --prompt-data.
        Other args are the same as Dataset.

    Returns:
        Dataset instance.
    """
    if isinstance(paths, list):
        if len(paths) != 1:
            raise ValueError(f"Only single-path datasets are supported, got {len(paths)} paths.")
        paths = paths[0]

    return Dataset(
        path=paths,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        prompt_key=prompt_key,
        multimodal_keys=multimodal_keys,
        label_key=label_key,
        tool_key=tool_key,
        metadata_key=metadata_key,
        seed=seed,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
