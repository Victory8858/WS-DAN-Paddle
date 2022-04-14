"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: 数据集构建
"""

from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset

import config


def getDataset(target_dataset, resize):
    if target_dataset == 'aircraft':
        return AircraftDataset(mode='train', resize=resize), AircraftDataset(mode='val', resize=config.image_size)
    elif target_dataset == 'bird':
        return BirdDataset(mode='train', resize=resize), BirdDataset(mode='val', resize=config.image_size)
    elif target_dataset == 'car':
        return CarDataset(mode='train', resize=resize), CarDataset(mode='val', resize=config.image_size)
    else:
        raise ValueError('No Dataset {}'.format(target_dataset))
