"""
- @Author: GaoDing
- @Date: 2022/04/14 10:00
- @Description: 模型训练
"""

import os
import time
import logging
import argparse
from tqdm import tqdm
import paddle
from datasets import getDataset
from paddle.io import DataLoader
from models.wsdan import WSDAN
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, batch_augment
import paddle.nn.functional as F


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bird", type=str,
                        help="Which dataset you want to verify? bird, car, aircraft")
    parser.add_argument("--epochs", default=80, type=int, help="how many epochs you want to train")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--input_size", default=(448, 448), type=tuple)
    parser.add_argument("--beta", default=5e-2, type=float)
    parser.add_argument("--net_name", default='inception_mixed_6e', type=str, help="feature extractor")
    parser.add_argument("--num_attentions", default=32, type=int, help="number of attention maps")
    parser.add_argument("--save_dir", default="FGVC", type=str)
    parser.add_argument("--ckpt", default=False, type=bool, help="if you want continue to train use the exist model")
    args = parser.parse_args()

    return args


def train():
    # read the parameters
    args = getArgs()

    # logging config
    save_path = os.path.join(args.save_dir, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(
        filename=os.path.join(save_path, 'train.log'), filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    logging.info('Current Trainning Model: {}'.format(args.dataset))

    # read the dataset
    train_dataset, val_dataset = getDataset(args.dataset, args.input_size)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), \
                               DataLoader(val_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)
    num_classes = train_dataset.num_classes

    logging.info('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(dataset_name=args.dataset, train_num=len(train_dataset), val_num=len(val_dataset)))
    logging.info('Batch Size:[{0}], Train Batches:[{1}], Val Batches:[{2}]'.format(args.batch_size, len(train_loader), len(val_loader)))

    # loss and metric
    loss_container = AverageMeter(name='loss')
    raw_metric = TopKAccuracyMetric(topk=(1, 5))
    crop_metric = TopKAccuracyMetric(topk=(1, 5))
    drop_metric = TopKAccuracyMetric(topk=(1, 5))

    # define the network
    logs = {}
    if args.ckpt:
        pretrained = False
    else:
        pretrained = True
    net = WSDAN(num_classes=num_classes, num_attentions=args.num_attentions, net_name=args.net_name, pretrained=pretrained)
    feature_center = paddle.zeros(shape=[num_classes, args.num_attentions * net.num_features])

    # Optimizer
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=2, gamma=0.9)
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9, weight_decay=1e-5, parameters=net.parameters())

    # 可以加载预先保存的模型以及参数
    if args.ckpt:
        net_state_dict = paddle.load(os.path.join(args.save_dir, args.dataset, args.dataset) + "_model.pdparams")
        optim_state_dict = paddle.load(os.path.join(args.save_dir, args.dataset, args.dataset) + "_model.pdopt")
        net.set_state_dict(net_state_dict)
        optimizer.set_state_dict(optim_state_dict)

    # loss function
    cross_entropy_loss = paddle.nn.CrossEntropyLoss()
    center_loss = CenterLoss()

    max_val_acc = 0  # 最好的精度
    for epoch in range(args.epochs):
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.get_lr()
        logging.info('Epoch {:03d}, lr= {:g}'.format(epoch + 1, optimizer.get_lr()))
        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, args.epochs))

        # metrics initialization
        loss_container.reset()
        raw_metric.reset()
        crop_metric.reset()
        drop_metric.reset()

        # begin training
        start_time = time.time()
        net.train()
        for i, (X, y) in enumerate(train_loader):
            optimizer.clear_grad()
            y_pred_raw, feature_matrix, attention_map = net(X)

            # Update Feature Center
            feature_center_batch = F.normalize(feature_center[y], axis=-1)
            feature_center[y] += args.beta * (feature_matrix.detach() - feature_center_batch)

            # Attention Cropping
            with paddle.no_grad():
                crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

            # crop images forward
            y_pred_crop, _, _ = net(crop_images)

            # Attention Dropping
            with paddle.no_grad():
                drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

            # drop images forward
            y_pred_drop, _, _ = net(drop_images)

            # loss
            batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                         cross_entropy_loss(y_pred_crop, y) / 3. + \
                         cross_entropy_loss(y_pred_drop, y) / 3. + \
                         center_loss(feature_matrix, feature_center_batch)

            # backward
            batch_loss.backward()
            optimizer.step()

            with paddle.no_grad():
                epoch_loss = loss_container(batch_loss.item())
                epoch_raw_acc = raw_metric(y_pred_raw, y)
                epoch_crop_acc = crop_metric(y_pred_crop, y)
                epoch_drop_acc = drop_metric(y_pred_drop, y)

            batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Crop Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
                epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
                epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        # end of this epoch
        logs['train_{}'.format(loss_container.name)] = epoch_loss
        logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
        logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
        logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
        logs['train_info'] = batch_info
        end_time = time.time()

        # write log for this epoch
        logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))

        # 学习率每两个epoch衰减一次
        scheduler.step()

        ## 验证
        net.eval()
        loss_container.reset()
        raw_metric.reset()
        start_time = time.time()
        net.eval()
        with paddle.no_grad():
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
        end_time = time.time()
        batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
        pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))
        # write log for this epoch
        logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
        logging.info('')
        net.train()
        pbar.close()

        # 模型保存，保存精度最高的模型
        if (epoch_acc[0] > max_val_acc):
            max_val_acc = epoch_acc[0]
            paddle.save(net.state_dict(), os.path.join(args.save_dir, args.dataset, args.dataset) + "_model.pdparams")
            paddle.save(optimizer.state_dict(), os.path.join(args.save_dir, args.dataset, args.dataset) + "_model.pdopt")


if __name__ == "__main__":
    train()
