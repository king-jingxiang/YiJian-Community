# -*- coding: utf-8 -*-
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is specially provided for Global AI Offensive and Defensive Challenge Track 1：Vaccination for Text-to-Image Generative Models (https://tianchi.aliyun.com/competition/entrance/532268/information?lang=en-us)

import torch
import numpy as np
import pandas as pd
from diffusers import KolorsPipeline, FluxPipeline
from typing import Tuple
from datasets import Dataset
from PIL import Image
from yijian_community.data import load_data, save_data
from yijian_community.model import HFTxt2ImgInfer
from yijian_community.defense import ThuCoaiShieldLM, InternVL2ImageDefense


class Txt2ImgAttackPipeline:

    def __init__(
        self,
        text_defense_model: str = "thu-coai/ShieldLM-7B-internlm2",
        txt2img_zh_model: str = "Kwai-Kolors/Kolors-diffusers",
        txt2img_en_model: str = "black-forest-labs/FLUX.1-schnell",
        image_defense_model: str = "OpenGVLab/InternVL2-2B",
    ) -> None:
        self.text_defense = ThuCoaiShieldLM(model_path=text_defense_model)
        self.txt2img_zh = HFTxt2ImgInfer(model_path=txt2img_zh_model, pipeline=KolorsPipeline, variant="fp16")
        self.txt2img_en = HFTxt2ImgInfer(model_path=txt2img_en_model, pipeline=FluxPipeline, torch_dtype=torch.bfloat16)
        self.image_defense = InternVL2ImageDefense(model_path=image_defense_model)

    def attack_data(self, data: str, lang: str = "zh") -> Tuple[int, Image.Image, int]:
        text_defense_res = self.text_defense.infer_data(data, lang=lang)
        if lang == "zh":
            img = self.txt2img_zh.infer_data(data, guidance_scale=5.0, num_inference_steps=50)
        elif lang == "en":
            img = self.txt2img_en.infer_data(data, guidance_scale=0.0, num_inference_steps=5, max_sequence_length=256)
        else:
            raise ValueError(f"Unsupported lang! lang can be 'zh' or 'en', but {lang} found!")

    def attack_dataset(
        self, dataset: Dataset, target_column: str = "prompt_zh", lang: str = "zh", batch_size: int = 100
    ) -> Dataset:
        text_defense_res_dataset = self.text_defense.infer_dataset(
            dataset, target_column=target_column, lang=lang, batch_size=batch_size
        )
