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


from abc import ABC, abstractmethod
from datasets import Dataset


# Base class for inference
class Infer(ABC):

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def infer_data(self, data: str):
        """inference on one data

        Args:
            data (str): the prompt text
        """
        pass

    @abstractmethod
    def infer_dataset(self, dataset: Dataset) -> Dataset:
        """inference on one datasets.Dataset

        Args:
            dataset (Dataset): evaluation dataset

        Returns:
            Dataset: result dataset
        """
        pass
