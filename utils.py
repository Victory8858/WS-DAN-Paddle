"""
- @Author: GaoDing
- @Date: 2022/04/10 20:00
- @Description: 一些工具
"""

from paddle.vision.transforms import transforms
import paddle.nn as nn
import paddle
import numpy as np
import random
import config


# Center Loss for Attention Regularization
class CenterLoss(paddle.nn.Layer):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.shape[0]


# Metric
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.shape[0]
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        target = paddle.reshape(target, [1, -1])
        correct = pred.equal(target.expand_as(pred))

        for i, k in enumerate(self.topk):
            temp = paddle.reshape(correct[:k], [-1])
            # correct_k = correct[:k].reshape(-1).float().sum(0)
            correct_k = temp.sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


# Callback
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
                (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, paddle.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            # for key in state_dict.keys():
            #     state_dict[key] = state_dict[key].cpu()
            #
            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                #     feature_center = feature_center.cpu()

                paddle.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)

            else:
                paddle.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.shape

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='BILINEAR') >= theta_c
            nonzero_indices = paddle.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                nn.functional.interpolate(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW), mode='BILINEAR'))
        crop_images = paddle.concat(crop_images, axis=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='BILINEAR') < theta_d)
        drop_masks = paddle.concat(drop_masks, axis=0)
        drop_images = images * drop_masks
        # plt.imshow(drop_images[0][0])
        # plt.imshow(np.array(drop_images[0]).transpose([1,2,0]))  #彩色
        # plt.show()
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


# transform in dataset
def getTransform(resize, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=32. / 255, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif mode == 'val' and config.target_dataset == 'bird':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=config.image_size),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
