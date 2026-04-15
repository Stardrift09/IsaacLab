# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: Pick up the object.

Terminates when the object is grasped and lifted above basket height.
Initial condition: robot starts from ground (start_in_air=False).
"""

from __future__ import annotations

import os
import numpy as np
import torch

from isaaclab.utils import configclass
from .test_pick_it_up import TestPickItUp, TestPickItUpCfg


@configclass
class PickItUpCfg(TestPickItUpCfg):
    camera_sensor_record: bool = False
    start_in_air: bool = False  # always start from ground for pick-up stage


class PickItUp(TestPickItUp):
    """Pick-up stage: terminate when the object is grasped and lifted."""

    def _get_dones(self):
        # Run parent logic to update all intermediate values
        # (self.high_enough, self.grasped, self.small_rotation, etc.)
        _, truncated = super()._get_dones()

        # Terminate when object is grasped AND lifted above basket top
        terminated = self.high_enough & self.grasped & self.small_rotation
        return terminated, truncated

    def get_feedback_from_vlm(self, output_dir) -> str:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        use_cpu = True
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/DeepCaption-VLA-7B",
            torch_dtype="auto",
            device_map="cpu" if use_cpu else "auto",
        )
        processor = AutoProcessor.from_pretrained("prithivMLmods/DeepCaption-VLA-7B")

        num_samples = 6
        all_files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".png") and f.startswith("rgb_")],
            key=lambda f: int(f.split("_")[1]),
        )
        if len(all_files) > num_samples:
            indices = np.linspace(0, len(all_files) - 1, num_samples, dtype=int)
            sampled_files = [all_files[i] for i in indices]
        else:
            sampled_files = all_files

        images_content = [
            {"type": "image", "image": os.path.join(output_dir, f)} for f in sampled_files
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    *images_content,
                    {
                        "type": "text",
                        "text": (
                            "Write a single caption describing the full sequence.\n"
                            "Your caption MUST cover these four points in order:\n"
                            "1. How the gripper approaches the object (from above or sideways).\n"
                            "2. Whether the gripper fingers close around the object at any point.\n"
                            "3. Whether the object is lifted above the table.\n"
                            "4. In the final frame: is the object still held or has it been dropped?\n\n"
                            "Caption:"
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if not use_cpu:
            inputs = inputs.to("cuda")
        generated_ids = model.generate(
            **inputs, max_new_tokens=300, repetition_penalty=1.1
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text
