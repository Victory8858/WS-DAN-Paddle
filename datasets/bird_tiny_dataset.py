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
- @Description: Brid 数据集构建
"""

import os
from PIL import Image
from paddle.io import Dataset
from utils import getTransform
import numpy as np

DATAPATH = "datasets/CUBTINY"


class BirdTinyDataset(Dataset):
    def __init__(self, mode='train', resize=(448, 448)):
        super(BirdTinyDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.image_idx = []
        self.image_path = {}
        self.image_label = {}
        self.num_classes = 5  # 共有5类

        # 从 image_class_labels.txt 获取图像标签
        if self.mode == 'train':
            file_name = "train.txt"
        else:
            file_name = "test.txt"

        with open(os.path.join(DATAPATH, file_name)) as f:
            for line in f.readlines():
                img_name, label = line.strip().split(',')
                self.image_label[img_name] = int(label)
                self.image_idx.append(img_name)

        # 图像预处理（resize，明暗度变换）等
        self.transform = getTransform(self.resize, self.mode)

    def __getitem__(self, item):
        current_image_name = self.image_idx[item]
        image = Image.open(os.path.join(DATAPATH, current_image_name)).convert('RGB')  # CHW
        image = self.transform(image)
        label = np.int64(self.image_label[current_image_name] - 1)

        return image, label  # 标签应该从0开始

    def __len__(self):
        return len(self.image_idx)


if __name__ == '__main__':
    train_dataset = BirdTinyDataset(mode='test')
    print(len(train_dataset))
    import matplotlib.pyplot as plt
    for i in range(5):
        raw_image, image_label = train_dataset[i]
        plt.imshow(raw_image[0])
        plt.show()
        print(raw_image.shape, image_label)
