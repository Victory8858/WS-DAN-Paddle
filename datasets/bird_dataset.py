"""
- @Author: GaoDing
- @Date: 2022/04/08 22:00
- @Description: CUB-200-2011 Brid 数据集构建
"""

import os
from PIL import Image
from paddle.io import Dataset
from utils import getTransform
import numpy as np

DATAPATH = "E:\\dataset\\Fine-grained\\CUB_200_2011"  # My Dataset Path


class BirdDataset(Dataset):
    def __init__(self, mode='train', resize=(448, 448)):
        super(BirdDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.image_idx = []
        self.image_path = {}
        self.image_label = {}
        self.num_classes = 200  # 共有200类(0-199)

        # 从 images.txt 获取图像索引号以及路径
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                idx, img_path = line.strip().split(' ')
                self.image_path[idx] = img_path

        # 从 image_class_labels.txt 获取图像标签
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                idx, label = line.strip().split(' ')
                self.image_label[idx] = int(label)

        # 从 train_test_split.txt 获取训练、测试图片索引存放到 self.image_id 列表
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                current_image_idx, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)
                if self.mode == 'train' and is_training_image:
                    self.image_idx.append(current_image_idx)
                if self.mode in ('val', 'test') and not is_training_image:
                    self.image_idx.append(current_image_idx)

        # 图像预处理（resize，明暗度变换）等
        self.transform = getTransform(self.resize, self.mode)

    def __getitem__(self, item):
        current_image_idx = self.image_idx[item]
        image = Image.open(os.path.join(DATAPATH, 'images', self.image_path[current_image_idx])).convert('RGB')  # CHW
        image = self.transform(image)
        label = np.int64(self.image_label[current_image_idx] - 1)

        return image, label  # 标签应该从0开始

    def __len__(self):
        return len(self.image_idx)


if __name__ == '__main__':
    train_dataset = BirdDataset(mode='test')
    print(len(train_dataset))
    for i in range(10):
        raw_image, image_label = train_dataset[i]
        print(raw_image.shape, image_label)
