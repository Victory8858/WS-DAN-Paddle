"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: AircraftDataset 数据集构建
"""

import os
import numpy as np
from PIL import Image
from paddle.io import Dataset
from utils import getTransform
import dataset_path_config

DATAPATH = dataset_path_config.aircraft_dataset_path

FILENAME_LENGTH = 7


class AircraftDataset(Dataset):
    def __init__(self, mode='train', resize=(448, 448)):
        super(AircraftDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.image_idx = []
        variants_dict = {}
        with open(os.path.join(DATAPATH, 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                variants_dict[line.strip()] = idx
        self.num_classes = len(variants_dict)

        if mode == 'train':
            list_path = os.path.join(DATAPATH, 'images_variant_trainval.txt')
        else:
            list_path = os.path.join(DATAPATH, 'images_variant_test.txt')

        self.images = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                fname_and_variant = line.strip()
                self.images.append(fname_and_variant[:FILENAME_LENGTH])
                self.labels.append(np.int64(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]]))

        self.transform = getTransform(self.resize, self.mode)

    def __getitem__(self, item):
        image = Image.open(os.path.join(DATAPATH, 'images', '%s.jpg' % self.images[item])).convert('RGB')
        image = self.transform(image)

        return image, self.labels[item]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_dataset = AircraftDataset(mode='test')
    print(len(train_dataset))
    for i in range(1):
        raw_image, image_label = train_dataset[i]
        print(raw_image.shape, image_label)
        plt.imshow(raw_image[0])
        plt.show()
