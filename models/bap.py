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
        feature_matrix = paddle.einsum('imjk,injk->imn', attentions, features) / float(H * W)
        feature_matrix = paddle.reshape(feature_matrix, [B, -1])
        # sign-sqrt
        feature_matrix = paddle.sign(feature_matrix) * paddle.sqrt(paddle.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, axis=-1)
        return feature_matrix
