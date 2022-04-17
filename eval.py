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
- @Description: 模型测试
"""

import os
import logging
import argparse
import paddle
from datasets import getDataset
from paddle.io import DataLoader
from models.wsdan import WSDAN
from utils import AverageMeter, TopKAccuracyMetric, batch_augment


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="car", type=str, help="Which dataset you want to verify? bird, car, aircraft, bird_tiny")
    parser.add_argument("--test-log-path", default="FGVC/", type=str)
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--input-size", default=(448, 448), type=tuple)
    parser.add_argument("--net-name", default='inception_mixed_6e', type=str, help="feature extractor")
    parser.add_argument("--num-attentions", default=32, type=int, help="number of attention maps")
    args = parser.parse_args()

    return args


def val():
    # read the parameters
    args = getArgs()

    # logging config
    logging.basicConfig(filename=os.path.join(args.test_log_path + args.dataset, 'test.log'), filemode='w',
                        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    logging.info('Current Testing Model: {}'.format(args.dataset))

    # read the dataset
    train_dataset, val_dataset = getDataset(args.dataset, args.input_size)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # output the dataset info
    logging.info('Dataset Name:{dataset_name}, Val:[{val_num}]'.format(dataset_name=args.dataset, train_num=len(train_dataset), val_num=len(val_dataset)))
    logging.info('Batch Size:[{0}], Train Batches:[{1}], Val Batches:[{2}]'.format(args.batch_size, len(train_loader), len(val_loader)))

    # loss and metric
    loss_container = AverageMeter(name='loss')
    raw_metric = TopKAccuracyMetric(topk=(1, 5))
    num_classes = train_dataset.num_classes

    # load the network and parameters
    net = WSDAN(num_classes=num_classes, num_attentions=args.num_attentions, net_name=args.net_name, pretrained=False)
    if args.dataset == 'bird':
        net_state_dict = paddle.load("FGVC/bird/bird_model.pdparams")
    if args.dataset == 'aircraft':
        net_state_dict = paddle.load("FGVC/aircraft/aircraft_model.pdparams")
    if args.dataset == 'car':
        net_state_dict = paddle.load("FGVC/car/car_model.pdparams")
    if args.dataset == 'bird_tiny':
        net_state_dict = paddle.load("FGVC/bird_tiny/bird_tiny_model.pdparams")
    net.set_dict(net_state_dict)
    net.eval()

    # loss function
    cross_entropy_loss = paddle.nn.CrossEntropyLoss()

    # start to val
    logs = {}
    for i, (X, y) in enumerate(val_loader):
        # Raw Image
        y_pred_raw, _, attention_map = net(X)

        # Object Localization and Refinement
        crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = net(crop_images)

        # Final prediction
        y_pred = (y_pred_raw + y_pred_crop) / 2.

        # loss
        batch_loss = cross_entropy_loss(y_pred, y)
        epoch_loss = loss_container(batch_loss.item())

        # metrics: top-1,5 error
        epoch_acc = raw_metric(y_pred, y)

    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    logging.info(batch_info)
    print(batch_info)


if __name__ == '__main__':
    val()
