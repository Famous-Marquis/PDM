# Copyright 2025 Beijing University of Posts and Telecommunications
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers

from metrics import FD_calculator


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


# todo: 降低扩散难度
'''
编码器：(L,)->(N,N), where N = lower integer of √L   
解码器:(L,)<-(N,N)

扩散模型网络，深层通道的卷积，提高序列预测维度。channel unsqueeze 1->16->64->16->1 squeeze
'''


class FCMean(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super(FCMean, self).__init__(name='FC')
        self.d_model = d_model
        self._layers = []
        # self._layers.append(keras.Input(shape=(d_model,)))
        # self.encoder = layers.Dense(units=model_struct[0], name='encoder')
        # self.encoder_activation = layers.LeakyReLU()
        # self.decoder = layers.Dense(units=d_model, name='decoder')
        # self.decoder_norm = layers.LayerNormalization()
        # self.time_embedding = TimeEmbedding(d_model)
        self.time_embedding = layers.Embedding(input_dim=1, output_dim=d_model)
        # todo: 考虑调整Leaky ReLU的alpha(可参照GAN的网络结构)
        for units in model_struct:
            self._layers.append(layers.Dense(units,
                                             kernel_regularizer=regularizers.l2(0.01),
                                             kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                          stddev=0.05)))
            self._layers.append(layers.LayerNormalization())
            self._layers.append(layers.LeakyReLU())
            # self._layers.append(layers.Embedding(input_dim=1, output_dim=units))
            self._layers.append(layers.Dropout(0.2))

        self._layers.append(layers.Dense(d_model,
                                         kernel_regularizer=regularizers.l2(0.01),
                                         kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                                      stddev=0.05)))
        self._layers.append(layers.LayerNormalization())
        self._layers.append(layers.Dropout(0.2))
        # self._layers.append(layers.LeakyReLU())

        # self.attn=_layers.Attention()
        # self.d5 = _layers.Dropout(0.2)

    def call(self, x, t):
        # y = self.encoder(x)
        emb = self.time_embedding(t)
        y = x + emb
        # y = self.encoder_activation(y)
        for layer in self._layers:
            if isinstance(layer, layers.Embedding):
                emb = layer(t)
                y = y + emb
            else:
                y = layer(y)
        # y = self.decoder(y)
        # y = self.decoder_norm(y)

        return y


class FCCov(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super(FCCov, self).__init__(name='FCCov')
        self.d_model = d_model
        self._layers = []
        # self._layers.append(keras.Input(shape=(d_model,)))
        self.time_embedding = layers.Embedding(input_dim=1, output_dim=d_model)
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


class AutoEncoder(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super().__init__()
        self._layers = []
        self.d_model = d_model
        self.model_struct = model_struct
        self.init()

    def init(self):
        self._layers = []
        for units in self.model_struct:
            self._layers.append(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01), ))
            self._layers.append(layers.LayerNormalization())
            self._layers.append(layers.LeakyReLU())
            self._layers.append(layers.Dropout(0.2))
        self._layers = self._layers[:-1]

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class AutoDecoder(keras.Model):
    def __init__(self, d_model, model_struct: List,conv_struct:List):
        super().__init__()
        self._layers = []
        self.d_model = d_model
        self.model_struct = model_struct
        self.conv_struct = conv_struct
        self.init()

    def init(self):
        for channel in self.conv_struct:
            self._layers.append(layers.Conv1D(channel,5,2,padding='same'))
            self._layers.append(layers.BatchNormalization())
            self._layers.append(layers.LeakyReLU())
            self._layers.append(layers.MaxPool1D())
            self._layers.append(layers.Dropout(0.2))
        # for units in reversed(self.model_struct):
        #     self._layers.append(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01), ))
        #     self._layers.append(layers.LayerNormalization())
        #     self._layers.append(layers.LeakyReLU())
        #     self._layers.append(layers.Dropout(0.2))
        self._layers = self._layers[:-1]

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class Discriminator(keras.Model):
    def __init__(self, d_model, model_struct: List,conv_struct:List):
        super().__init__()
        self._layers = []
        self.d_model = d_model
        self.model_struct = model_struct
        self.init()

    def init(self):
        for units in self.model_struct:
            self._layers.append(layers.Dense(units, kernel_regularizer=regularizers.l2(0.01), ))
            self._layers.append(layers.LayerNormalization())
            self._layers.append(layers.LeakyReLU())
            self._layers.append(layers.Dropout(0.2))
        self._layers.append(layers.Dense(1, kernel_regularizer=regularizers.l2(0.01), ))
        self._layers.append(layers.Activation('sigmoid'))

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class VAEMonitor(keras.callbacks.Callback):
    def __init__(self, real_z):
        self.real_z = real_z
        self.FD_calculator = FD_calculator(real_z)

    def on_epoch_end(self, epoch, logs=None):
        z = self.model.encoder(self.real_z)
        z_hat = self.model.decoder(z).numpy()
        fd = self.FD_calculator.frechet_distance(z_hat)
        logs['val_fd'] = fd
        print('FD=', fd)
        # print("FD:",fd)


class VAE(keras.Model):
    def __init__(self, d_model, model_struct: List):
        super().__init__()
        self.loss_fn_vae = None
        self.loss_fn_disc = None
        self.vae_optimizer = None
        self.d_optimizer = None
        self.d_model = d_model
        self.model_struct = model_struct
        self.discriminator = Discriminator(d_model, model_struct)
        self.encoder = AutoEncoder(d_model, model_struct)
        self.decoder: keras.Model = AutoDecoder(d_model, model_struct)
        self.loss_tracker_vae = keras.metrics.Mean(name='loss_d')
        self.loss_tracker_d = keras.metrics.Mean(name='loss_disc')
        # self.loss_tracker_g = keras.metrics.Mean(name='loss_g')
        self.latent_dim = model_struct[-1]

    def compile(self, d_optimizer, vae_optimizer,
                loss_fn_disc=keras.losses.BinaryCrossentropy(from_logits=False,
                                                             label_smoothing=0.05),
                loss_fn_vae=keras.losses.MeanSquaredError(), ):
        super().compile()
        self.d_optimizer = d_optimizer
        # self.encoder_optimizer = encoder_optimizer
        self.vae_optimizer = vae_optimizer
        self.loss_fn_disc = loss_fn_disc
        self.loss_fn_vae = loss_fn_vae
        # self.loss_fn_decoder = loss_fn_decoder

    @property
    def metrics(self):
        return [self.loss_tracker_d, self.loss_tracker_vae]

    @tf.function
    def train_step(self, real_x):
        batch = tf.shape(real_x)[0]
        z = tf.random.normal(shape=(batch, self.latent_dim))
        generated_x = self.decoder(z)
        labels = tf.concat([tf.ones((batch, 1)), tf.zeros((batch, 1))], axis=0)
        coeffs = tf.concat([real_x, generated_x], axis=0)
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # 1 训练判别器
        with tf.GradientTape() as tape:
            predictions = self.discriminator(coeffs)
            d_loss = self.loss_fn_disc(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.loss_tracker_d.update_state(d_loss)
        # 2训练编码器&训练解码器
        with tf.GradientTape() as tape:
            # 正确的流程是：先编码，后解码
            z = self.encoder(real_x, training=True)
            x_hat = self.decoder(z, training=True)
            predictions = self.discriminator(x_hat, training=True)
            # 重建损失（比如 L1）
            loss_recon = self.loss_fn_vae(x_hat, real_x)
            # 生成器的对抗损失（假样本希望判别为真，标签为1）
            loss_adv = self.loss_fn_disc(tf.ones_like(predictions), predictions)
            # 总损失
            loss = loss_recon + 0.5*loss_adv
        # 收集所有需要更新的参数
        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        # 求梯度
        grads = tape.gradient(loss, train_vars)
        # 更新权重
        self.vae_optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker_vae.update_state(loss)

        return {"d_loss": d_loss, "vae_loss": loss}


class AutoEncoderHelper:
    def __init__(self, d_model, model_struct: List):
        self.d_model = d_model
        self.model_struct = model_struct
        self.discriminator = Discriminator(d_model, model_struct)
        self.encoder = AutoEncoder(d_model, model_struct)
        self.decoder = AutoDecoder(d_model, model_struct)

    def train(self):
        ...


if __name__ == '__main__':
    vae = VAE(231, [231, 512, 512,231 // 2])
    vae.compile(vae_optimizer=keras.optimizers.Adam(lr=0.0001),
                d_optimizer=keras.optimizers.Adam(lr=0.0001), )
    x_train = np.load("./Datasets/train-param1-5000.npy").astype(np.float32)
    # x_train=tf.Tensor(x_train,dtype=tf.float32)
    tf_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(64)
    vae.fit(tf_ds, epochs=500, shuffle=True, callbacks=[VAEMonitor(x_train)])
