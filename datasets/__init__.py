# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: 数据集构建
"""

from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .bird_tiny_dataset import BirdTinyDataset


def getDataset(target_dataset, resize):
    if target_dataset == 'aircraft':
        return AircraftDataset(mode='train', resize=resize), AircraftDataset(mode='val', resize=resize)
    elif target_dataset == 'bird':
        return BirdDataset(mode='train', resize=resize), BirdDataset(mode='val', resize=resize)
    elif target_dataset == 'car':
        return CarDataset(mode='train', resize=resize), CarDataset(mode='val', resize=resize)
    elif target_dataset == 'bird_tiny':
        return BirdTinyDataset(mode='train', resize=resize), BirdTinyDataset(mode='val', resize=resize)
    else:
        raise ValueError('No Dataset {}'.format(target_dataset))
