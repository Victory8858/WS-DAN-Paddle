"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: 输入图片进行预测类别
"""

import argparse
import paddle
from datasets import getDataset
from models.wsdan import WSDAN
from utils import batch_augment
import numpy as np
import matplotlib.pyplot as plt

ckpt_path = {
    'bird': "FGVC/bird/bird_model.pdparams",
    'car': "FGVC/car/car_model.pdparams",
    'aircraft': "FGVC/aircraft/aircraft_model.pdparams"
}

num_images = {
    'bird': 5794,
    'car': 8041,
    'aircraft': 3333
}


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bird", type=str,  # 想更换不同的数据集测试，仅仅更改这里即可
                        help="Which dataset you want to verify? bird, car, aircraft")
    parser.add_argument("--input_size", default=(448, 448), type=tuple)
    parser.add_argument("--net_name", default='inception_mixed_6e', type=str, help="feature extractor")
    parser.add_argument("--num_attentions", default=32, type=int, help="number of attention maps")
    args = parser.parse_args()

    return args


def predicted():
    # read the parameters
    args = getArgs()

    _, test_dataset = getDataset(args.dataset, args.input_size)
    image, real_label = test_dataset[np.random.randint(0, num_images[args.dataset])]
    plt.imshow(image[0])
    plt.show()
    image = np.expand_dims(image, axis=0)
    image = paddle.to_tensor(image)

    # load the network and parameters
    num_classes = test_dataset.num_classes
    model = WSDAN(num_classes=num_classes, num_attentions=args.num_attentions, net_name=args.net_name, pretrained=False)
    net_state_dict = paddle.load(ckpt_path[args.dataset])
    model.set_dict(net_state_dict)
    model.eval()

    # Raw Image
    y_pred_raw, _, attention_map = model(image)

    # Object Localization and Refinement
    crop_images = batch_augment(image, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
    y_pred_crop, _, _ = model(crop_images)

    # Final prediction
    y_pred = (y_pred_raw + y_pred_crop) / 2.
    print('Object:{}\nReal Category:{}\nPredict Category:{} '.format(args.dataset, real_label, np.argmax(y_pred)))


if __name__ == '__main__':
    predicted()
