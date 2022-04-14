import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# Bilinear Attention Pooling(原版)
# class BAP(paddle.nn.Layer):
#     def __init__(self, **kwargs):
#         super(BAP, self).__init__()
#
#     def forward(self, feature_maps, attention_maps):
#         feature_shape = feature_maps.shape  ## 12*768*26*26*
#         attention_shape = attention_maps.shape  ## 12*num_parts*26*26
#         # print(feature_shape,attention_shape)
#         phi_I = paddle.einsum('imjk,injk->imn', (attention_maps, feature_maps))  ## 12*32*768
#         phi_I = paddle.divide(phi_I, float(attention_shape[2] * attention_shape[3]))
#         phi_I = paddle.multiply(paddle.sign(phi_I), paddle.sqrt(paddle.abs(phi_I) + 1e-12))
#         phi_I = phi_I.reshape(feature_shape[0], -1)
#         raw_features = nn.functional.normalize(phi_I, axis=-1)  ##12*(32*768)
#         pooling_features = raw_features * 100
#         # print(pooling_features.shape)
#         return raw_features, pooling_features


class BAP(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super(BAP, self).__init__()
        self.pool = None

    def forward(self, features, attentions):
        B, C, H, W = features.shape
        _, M, AH, AW = attentions.shape
        if AH != H or AW != W:
            F.interpolate(attentions, size=[H, W], mode="bilinear")
        feature_matrix = paddle.einsum('imjk,injk->imn', attentions, features) / float(H * W)
        feature_matrix = paddle.reshape(feature_matrix, [B, -1])
        # sign-sqrt
        feature_matrix = paddle.sign(feature_matrix) * paddle.sqrt(paddle.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, axis=-1)
        return feature_matrix


class ResizeCat(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super(ResizeCat, self).__init__()

    def forward(self, at1, at3, at5):
        N, C, H, W = at1.shape
        resized_at3 = nn.functional.interpolate(at3, size=[H, W])
        resized_at5 = nn.functional.interpolate(at5, size=[H, W])
        cat_at = paddle.concat((at1, resized_at3, resized_at5), axis=1)
        return cat_at


if __name__ == '__main__':
    # a = BAP()
    a = ResizeCat()
    # a1 = paddle.to_tensor(4, 3, 14, 14)
    # a3 = paddle.to_tensor(4, 5, 12, 12)
    # a5 = paddle.to_tensor(4, 9, 9, 9)
    a1 = paddle.rand((4, 3, 14, 14))
    a3 = paddle.rand((4, 5, 12, 12))
    a5 = paddle.rand((4, 9, 9, 9))
    ret = a(a1, a3, a5)
    print(ret.shape)
