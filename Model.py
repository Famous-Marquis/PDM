from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers

from aberration import ZERNIKE_NUMS


class TimeEmbedding(layers.Layer):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__(name='TimeEmbedding')
        self.d_model = d_model
        assert self.d_model % 2 == 0, "d_model must be divisible by 2"
        self.omega = 1 / (tf.math.pow(10000, tf.range(0, self.d_model, 2)
                                      / self.d_model))[None, :]
        self.omega = tf.cast(self.omega, tf.float32)

    def call(self, t, training=False, mask=None):
        t_dim = self.d_model // 2
        if len(t.shape) == 0:  # 如果 t 是标量
            t = tf.expand_dims(t, axis=0)  # 转换为1维张量
        t = t[:, None]  # 现在可以安全地执行切片操作
        # t = t[:, None]
        t = tf.cast(t, dtype=tf.float32)
        emb_even = tf.math.sin(t * self.omega)
        emb_odd = tf.math.cos(t * self.omega)
        assert list(emb_even.shape) == [t.shape[0], t_dim], "emb 形状不对"

        emb_even = tf.reshape(emb_even, [-1, t_dim])  # 确保形状为 [batch_size, t_dim]
        emb_odd = tf.reshape(emb_odd, [-1, t_dim])  # 确保形状为 [batch_size, t_dim]

        # 将 emb_even 和 emb_odd 组合成最终的嵌入向量
        emb = tf.stack([emb_even, emb_odd], axis=-1)  # 形状为 [batch_size, t_dim, 2]
        emb = tf.reshape(emb, [-1, self.d_model])  # 重塑为 [batch_size, d_model]
        return emb


class FCMean(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super(FCMean, self).__init__(name='FC')
        self.d_model = d_model
        self._layers = []
        # self._layers.append(keras.Input(shape=(d_model,)))
        self.time_embedding = layers.Embedding(input_dim=1,output_dim=d_model)
        # todo: 考虑调整Leaky ReLU的alpha(可参照GAN的网络结构)
        for units in model_struct:
            self._layers.append(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01),
                                             kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                          stddev=0.05)))
            self._layers.append(layers.LayerNormalization())
            self._layers.append(layers.LeakyReLU())
            self._layers.append(layers.Dropout(0.2))

        self._layers.append(layers.Dense(d_model, kernel_regularizer=regularizers.l2(0.01),
                                         kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                      stddev=0.05)))
        self._layers.append(layers.LayerNormalization())
        self._layers.append(layers.LeakyReLU())

        # self.attn=_layers.Attention()
        # self.d5 = _layers.Dropout(0.2)

    def call(self, x, t):
        emb = self.time_embedding(t)
        x = x + emb
        for layer in self._layers:
            x = layer(x)
        # y = self.d5(x)

        return x


class FCCov(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super(FCCov, self).__init__(name='FCCov')
        self.d_model = d_model
        self._layers = []
        # self._layers.append(keras.Input(shape=(d_model,)))
        self.time_embedding = layers.Embedding(input_dim=1,output_dim=d_model)
        for units in model_struct:
            self._layers.append(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01),
                                             kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                          stddev=0.005)))
            self._layers.append(layers.LayerNormalization())
            self._layers.append(layers.LeakyReLU())
            self._layers.append(layers.Dropout(0.2))

        self._layers.append(layers.Dense(d_model, kernel_regularizer=regularizers.l2(0.01),
                                         kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                      stddev=0.005)))

        self._layers.append(layers.LayerNormalization())
        self._layers.append(layers.LeakyReLU())
        # self.d5 = _layers.Dropout(0.2)

    def call(self, x, t):
        emb = self.time_embedding(t)
        x = x + emb
        for layer in self._layers:
            x = layer(x)
        # 裁剪，防止协方差为负
        y = tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)
        return y


if __name__ == '__main__':
    model = FCMean(d_model=ZERNIKE_NUMS)
    model_v = FCCov(d_model=ZERNIKE_NUMS)

    # 定义输入
    input_shape_x = ZERNIKE_NUMS
    input_shape_t = ()
    input_x = tf.random.uniform(shape=(64, ZERNIKE_NUMS,))
    input_t = tf.random.uniform(shape=(64,), maxval=100, minval=0, dtype=tf.int32)
    model(input_x, input_t)
    model_v(input_x, input_t)
    model.summary()
    model_v.summary()
