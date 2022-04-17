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
- @Description: BAP模型
"""

import paddle
import paddle.nn.functional as F


class BAP(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super(BAP, self).__init__()
        self.pool = None

    def forward(self, features, attentions):
        B, C, H, W = features.shape
        _, M, AH, AW = attentions.shape
        if AH != H or AW != W:
            F.interpolate(attentions, size=[H, W], mode="bilinear")

        # 此段程序用来替代einsum函数
        mat = []
        for i in range(B):  # batch 拆分
            cur_atm = attentions[i]  # 去除第一维
            cur_ftm = features[i]  # 去除第一维
            cur_atm = paddle.reshape(cur_atm, shape=[M, -1])  # 展开
            # print("cur_atm shape: ", cur_atm.shape)
            cur_ftm = paddle.reshape(cur_ftm, shape=[C, -1])  # 展开
            cur_ftm = paddle.transpose(cur_ftm, perm=[1, 0])  # 转置
            # print("cur_ftm shape: ", cur_ftm.shape)
            cur_feature_matrix = paddle.matmul(cur_atm, cur_ftm)  # 矩阵相乘，相当于点积
            mat.append(cur_feature_matrix)
        feature_matrix = paddle.stack(mat, axis=0) / float(H * W)

        # feature_matrix = paddle.einsum('imjk,injk->imn', attentions, features) / float(H * W)  # 动转静不能用
        feature_matrix = paddle.reshape(feature_matrix, [B, -1])

        # sign-sqrt
        feature_matrix = paddle.sign(feature_matrix) * paddle.sqrt(paddle.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, axis=-1)
        return feature_matrix
