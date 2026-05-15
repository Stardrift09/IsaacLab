# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2: Place the already-grasped object inside the basket.

Terminates when the object lands inside the basket (inside_site & low_enough).
Initial condition: robot starts mid-trajectory with object already grasped
(start_in_air=True).
"""

from __future__ import annotations

import os
import numpy as np
import torch

from isaaclab.utils import configclass
from .test_pick_it_up import TestPickItUp, TestPickItUpCfg


@configclass
class PlaceInBasketCfg(TestPickItUpCfg):
    camera_sensor_record: bool = False
    start_in_air: bool = False  # always start mid-air (object already grasped)


class PlaceInBasket(TestPickItUp):
    """Placement stage: start with object grasped, terminate when inside basket."""

    def _get_dones(self):
        # Run parent logic to update all intermediate values
        _, truncated = super()._get_dones()

        # Terminate when object is inside the basket and low enough
        terminated = self.inside_site & self.low_enough & self.high_enough_for_basket
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

        num_samples = 30
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
                        "text": """
Analyze the sequence of images showing a robot arm trying to place an object into a basket.
There is one coordinate frame attached to the object.

Answer each question using ONLY: (yes / no / unsure). Repeat the question before answering.

Q1: Is the object positioned above the basket opening at any point in the sequence?
A1: <yes/no/unsure>

Q2: Is the object released (gripper opens) while above the basket?
A2: <yes/no/unsure>

Q3: In the final frame, is the object inside the basket (below the basket rim and within its bounds)?
A3: <yes/no/unsure>

Q4: In the final frame, is the origin of the object coordinate frame located at the center of the gripper (which would indicate the object was NOT dropped)?
A4: <yes/no/unsure>

Reasoning: <brief explanation based on visible positions of object, gripper, and basket>
""",
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
            **inputs, max_new_tokens=300, repetition_penalty=1.3, no_repeat_ngram_size=3
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
