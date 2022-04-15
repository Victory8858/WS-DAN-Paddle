"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: 模型测试
"""

import os
import logging
import config
import paddle
from datasets import getDataset
from paddle.io import DataLoader
from models.wsdan import WSDAN
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, batch_augment

# which dataset you want to test
config.target_dataset = 'bird'  # it can be 'car', 'bird', 'aircraft'

# logging config
logging.basicConfig(
    filename=os.path.join("C:/Users/Victory/Desktop/WS-DAN-Paddle-Victory8858/FGVC/" + config.target_dataset + "/ckpt/",
                          'test.log'),
    filemode='w',
    format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
    level=logging.INFO)
logging.info('Current Testing Model: {}'.format(config.target_dataset))

# read the dataset
train_dataset, val_dataset = getDataset(config.target_dataset, config.input_size)
train_loader, val_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      num_workers=config.workers), \
                           DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                      num_workers=config.workers)
# output the dataset info
logging.info(
    'Dataset Name:{dataset_name}, Val:[{val_num}]'.format(dataset_name=config.target_dataset,
                                                          train_num=len(train_dataset),
                                                          val_num=len(val_dataset)))
logging.info('Batch Size:[{0}], Train Batches:[{1}], Val Batches:[{2}]'.format(config.batch_size, len(train_loader),
                                                                               len(val_loader)))

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))
num_classes = train_dataset.num_classes

# network
net = WSDAN(num_classes=num_classes, num_attentions=config.num_attentions, net_name=config.net_name,
            pretrained=False)
feature_center = paddle.zeros(shape=[num_classes, config.num_attentions * net.num_features])
if config.target_dataset == 'bird':
    net_state_dict = paddle.load("FGVC/bird/ckpt/bird_model.pdparams")
if config.target_dataset == 'aircraft':
    net_state_dict = paddle.load("FGVC/aircraft/ckpt/aircraft_model.pdparams")
if config.target_dataset == 'car':
    net_state_dict = paddle.load("FGVC/car/ckpt/car_model.pdparams")
net.set_state_dict(net_state_dict)
net.eval()

# loss function
cross_entropy_loss = paddle.nn.CrossEntropyLoss()
criterion = paddle.nn.CrossEntropyLoss()
center_loss = CenterLoss()

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
batch_info = 'Val Loss {:.4f}, Val Acc ({:.3f}, {:.3f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
print(batch_info)
