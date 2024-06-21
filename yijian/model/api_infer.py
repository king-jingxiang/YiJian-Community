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


from yijian.model.base_infer import Infer
from yijian.utils.util_func import console


class APIInfer(Infer):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        console.log(
            "This class will be implemented in the next update or you are welcomed to make a contribution."
        )


class OpenAITxt2TxtInfer(APIInfer):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        console.log("Reference: https://github.com/openai/openai-python")


class AnthropicTxt2TxtInfer(APIInfer):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        console.log("Reference: https://github.com/anthropics/anthropic-sdk-python")


class CohereTxt2TxtInfer(Infer):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        console.log("Reference: https://github.com/cohere-ai/cohere-python")
