import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.inception import inception_v3, BasicConv2d
from models.bap import BAP
import logging

EPSILON = 1e-12


class WSDAN(paddle.nn.Layer):
    def __init__(self, num_classes, num_attentions=32, net_name='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.num_attentions = num_attentions
        self.net_name = net_name

        # Network Initialization
        self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
        self.num_features = 768

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.num_attentions, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.num_attentions * self.num_features, self.num_classes, bias_attr=False)

        logging.info(
            'WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net_name,
                                                                                               self.num_classes,
                                                                                               self.num_attentions))

    def forward(self, x):
        batch_size = x.shape[0]

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net_name != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = paddle.sqrt(attention_maps[i].sum(axis=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, axis=0)
                k_index = np.random.choice(self.num_attentions, 2, p=attention_weights.numpy())
                atm = paddle.stack([attention_maps[i, k_index[0], :, :], attention_maps[i, k_index[1], :, :]])
                attention_map.append(atm)  # 16,32,3,3
            attention_map = paddle.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = paddle.mean(attention_maps, axis=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)
