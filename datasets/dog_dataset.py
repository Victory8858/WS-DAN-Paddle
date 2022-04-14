"""
- @Author: GaoDing
- @Date: 2022/04/08 22:00
- @Description: Dog 数据集构建
"""

import os

from PIL import Image
from paddle.io import Dataset
from utils import getTransform

# DATAPATH = '/home/aistudio/data/CUB-200-2011'       # AI_Studio
DATAPATH = "E:\\dataset\\Fine-grained\\dogs"          # My_3070


class DogDataset(Dataset):
    def __init__(self, mode='train', resize=(448, 448)):
        super(DogDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.image_idx = []
        self.image_path = {}
        self.image_label = {}

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.image_idx)