import os
import sys

import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from Train import frechet_distance
from config import MODEL_CONFIG
from metrics import FD_calculator
from plot_struct import plot_struct

print(sys.path)

# from sklearn.model_selection import train_test_split
from aberration import PhaseScreen, ZERNIKE_NUMS

import numpy as np

"""
# Jensen-Shannon Divergence
"""

"""
## Override `train_step`
"""


class GAN(keras.Model):
    """
    生成对抗网络（GAN）模型类，封装了判别器和生成器。

    参数:
    ----------------
    discriminator : keras.Model
        GAN中的判别器模型，用于区分真实数据和生成的假数据。
    generator : keras.Model
        GAN中的生成器模型，用于从潜在空间生成数据。
    latent_dim : int
        生成器输入的潜在向量的维度。
    test_set : numpy.ndarray
        用于评估和测试的一组真实数据。

    属性:
    -------------
    d_optimizer : keras.optimizers.Optimizer
        判别器模型的优化器。
    encoder_optimizer : keras.optimizers.Optimizer
        生成器模型的优化器。
    loss_fn : keras.losses.Loss
        用于训练GAN的损失函数。

    方法:
    ------------
    1.compile : 编译模型，设置优化器和损失函数。
    2.train_step : 进行单步训练，更新判别器和生成器的权重。

    """

    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.loss_fn = None
        self.d_optimizer = None
        self.g_optimizer = None
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.loss_tracker_d = keras.metrics.Mean(name='loss')
        self.loss_tracker_g = keras.metrics.Mean(name='loss')

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        """
        方法:
        ------------
        compile : 编译模型，设置优化器和损失函数。
        """
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def generate_samples(self, sample_num):
        batch_size = 100
        num_steps = sample_num // batch_size
        sample_num = num_steps * batch_size
        samples = tf.TensorArray(tf.float32, size=num_steps, dynamic_size=False)
        for step in range(num_steps):
            noise = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_samples = self.generator(noise)
            samples = samples.write(step, generated_samples)
        samples = samples.stack()
        samples = tf.reshape(samples, (sample_num, -1))
        return samples
    @tf.function
    def train_step(self, real_z: tf.Tensor):
        """
        进行单步训练，更新判别器和生成器的权重。

        参数:
        --------------
        real_z: 真实数据的输入张量，形状为(batch_size, ...)。

        返回:
        dict
            包含判别器和生成器损失的字典。

        说明:
        - 此方法用于训练GAN中的判别器和生成器模型。
        - 首先，训练判别器以区分真实数据和生成的假数据。
        - 然后，训练生成器以生成判别器认为是真实的数据。
        - 最后，可选地计算JS散度作为生成数据与真实数据分布差异的度量。
        """

        # 从潜在空间中采样随机点
        batch_size = tf.shape(real_z)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 通过生成器解码这些潜在向量以生成假图像
        generated_images = self.generator(random_latent_vectors)

        # 将生成的假图像与真实图像结合
        combined_images = tf.concat([generated_images, real_z], axis=0)

        # 为真实和假图像分配标签
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # 给标签添加随机噪声，提高训练的稳定性
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 训练判别器
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 再次从潜在空间中采样随机点用于生成器训练
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 为生成器生成的图像分配误导性标签，即全部标记为假图像
        misleading_labels = tf.zeros((batch_size, 1))

        # 训练生成器（注意：这里不更新判别器的权重）
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # 可选：更新JS散度度量（此部分代码被注释掉了）
        # random_latent_vectors = tf.random.normal(
        #     shape=(self.test_set.shape[0], self.latent_dim)
        # )
        # generated_zernike = self.generator(random_latent_vectors)
        # jsd = JSD(generated_zernike, self.test_set)
        self.loss_tracker_d.update_state(d_loss)
        self.loss_tracker_g.update_state(g_loss)
        # 返回判别器和生成器的损失
        return {"d_loss": d_loss, "g_loss": g_loss}

    @property
    def metrics(self):
        return [self.loss_tracker_d, self.loss_tracker_g]


"""
## Create a callback that periodically calculate JSD
"""


class GANMonitor(keras.callbacks.Callback):
    """
    用于监控GAN训练过程中生成数据的质量。

    参数:
    real_zernike : numpy.ndarray
        真实数据集，用于与生成的数据进行比较。
    latent_dim : int, optional
        潜在空间的维度，默认为128。

    属性:
    real_zernike : numpy.ndarray
        真实数据集。
    real_samples_num : int
        真实数据集中样本的数量。
    latent_dim : int
        潜在空间的维度。
    fds : list or numpy.ndarray
        存储每个epoch结束时计算的Frechet距离。
    model : keras.Model
        当前正在训练的模型，由Keras回调机制自动设置。

    方法:
    on_epoch_end : 在每个epoch结束时调用，用于评估生成器的性能。
    """

    def __init__(self, real_zernike, latent_dim=128):
        self.model: keras.Model  # 当前正在训练的模型，由Keras回调机制自动设置
        self.real_zernike = real_zernike
        self.real_samples_num = real_zernike.shape[0]
        self.latent_dim = latent_dim
        self.ckpt_path = MODEL_CONFIG["gan_checkpoint_path"]
        self.FD_calculator=FD_calculator(real_zernike)
        self.fds = []

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.real_samples_num, self.latent_dim)
        )
        generated_zernike = self.model.generator(random_latent_vectors)
        fd = frechet_distance(self.real_zernike,np.array(generated_zernike))
        self.fds.append(fd)
        logs["val_loss"] = fd
        print("epoch {} FD : {}".format(epoch, fd))

    def on_train_end(self, logs=None):
        # 样本可视化
        maxlen = min(500, self.real_zernike.shape[0])
        z = tf.random.normal(shape=(maxlen, self.latent_dim))
        generated_zernike = self.model.generator(z)
        generated_zernike = generated_zernike.numpy()
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 5))
        for i, ax in enumerate(axes.flat):
            bar = ax.imshow(generated_zernike[i][None, :], aspect='auto', cmap="viridis")
            fig.colorbar(bar, ax=ax, orientation='vertical')
            ax.set_yticks([])
        fig.suptitle('GAN generated samples')
        fig.tight_layout()
        plt.savefig("./SampledImgs/GAN_samples.png", dpi=300)
        plt.close(fig)
        # FD绘图
        fig1 = plt.figure()
        plt.plot(self.fds)
        plt.xlabel('epoch')
        plt.ylabel('GAN Frechet distance')
        plt.show()
        plt.close(fig1)

        # struct绘制
        Dphi_1d_mean_eval, Dphi_1d_std_eval,r_over_r0_eval = plot_struct(generated_zernike)
        Dphi_1d_mean, Dphi_1d_std,r_over_r0 = plot_struct(self.real_zernike[:maxlen])

        fig2 = plt.figure(figsize=(6, 4))
        plt.plot(r_over_r0, Dphi_1d_mean, 'bo-', label='batch_data $D_\phi(r)$')
        plt.plot(r_over_r0_eval, Dphi_1d_mean_eval, 'ro-', label='sampled $D_\phi(r)$')
        plt.xlabel(r'$r / r_0$')
        plt.ylabel(r'$D_\phi(r)$')
        plt.title('GAN structure function')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./SampledImgs/GAN_struct.png", dpi=300)
        plt.close(fig2)


"""
## build a gan trainer to repeat training process
"""


class GANHelper:
    """
    辅助类，用于构建和训练生成对抗网络（GAN）。

    属性:
    - zernike_dim: int
        Zernike多项式的阶数，用于定义相位屏幕的复杂度。
    - latent_dim: int
        潜在空间的维度，决定了GAN生成数据的多样性。
    - data_length: int
        单个数据集中数据点的长度。
    - scr: PhaseScreen
        相位屏幕类的一个实例，用于模拟大气湍流效应。
    - gan: GAN
        GAN模型实例，包含判别器和生成器网络。

    方法:
    - train: 训练GAN模型，使用给定的数据集。
    - repeat_train: 重复训练GAN模型，允许在每次迭代中更新数据。
    - build_generator: 构建用于生成数据的生成器模型。
    - build_discriminator: 构建用于区分真实和生成数据的判别器模型。
    - load_data: 加载用于训练的初始数据集。
    - update_data: 更新训练数据集，可能用于数据增强。
    - evaluate: 评估生成器在随机潜在向量上的性能。
    """

    def __init__(self,data_path):
        self.zernike_dim = ZERNIKE_NUMS
        self.latent_dim = 128
        self.data_length = 4000
        self.model_config = MODEL_CONFIG
        self.ckpt_path = self.model_config["gan_checkpoint_path"]
        self.data_path = data_path
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        self.scr = PhaseScreen()
        self.gan = GAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=self.latent_dim,
        )
        generator_initial_learning_rate = self.model_config["gan_learning_rate"]
        generator_decay_steps = 100
        generator_decay_rate = 0.9
        generator_lr_schedule = ExponentialDecay(
            generator_initial_learning_rate,
            decay_steps=generator_decay_steps,
            decay_rate=generator_decay_rate
        )

        # 配置判别器的学习率调度器
        discriminator_initial_learning_rate = self.model_config["gan_learning_rate"]
        discriminator_decay_steps = 100
        discriminator_decay_rate = 0.9
        discriminator_lr_schedule = ExponentialDecay(
            discriminator_initial_learning_rate,
            decay_steps=discriminator_decay_steps,
            decay_rate=discriminator_decay_rate
        )

        self.gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_lr_schedule),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=generator_lr_schedule),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        )

    def _load_weights(self):
        if self.model_config["GAN_load_weights"]:
            if os.path.exists(self.ckpt_path + ".index"):
                # if os.path.exists(self.model_checkpoint_path):
                try:
                    self.gan.load_weights(self.ckpt_path)
                except Exception as e:
                    print("Weights load failed, {}".format(e))
                else:
                    print("Weights load succeeded")
            else:
                print("No checkpoint found")

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.model_config["GAN_epochs"]
        batch_size = self.model_config["batch_size"]

        tf_dataset, data_matrix = self.load_data(batch_size)

        self._load_weights()

        callbacks = [GANMonitor(data_matrix), keras.callbacks.ModelCheckpoint(
            self.ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )]

        history = self.gan.fit(
            tf_dataset,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True,
        )

        return history

    def load_data(
            self, batch
    ):
        # prepare test batch_data
        data_matrix = np.load(self.data_path).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data_matrix)
        dataset = dataset.shuffle(buffer_size=10 * batch).batch(batch,
                                                                drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        return dataset, data_matrix

    def build_discriminator(self):
        discriminator = keras.Sequential(
            [
                keras.Input(shape=(self.zernike_dim,)),
                layers.Dense(512),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(256),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(1),
            ],
            name="discriminator",
        )
        discriminator.summary()
        return discriminator

    def build_generator(
            self,
    ):
        generator = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                # We want to generate 128 coefficients to reshape into a 7x7x128 map
                layers.Dense(256),
                layers.LeakyReLU(alpha=0.2),
                # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.Dense(256),
                layers.LeakyReLU(alpha=0.2),
                # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.Dense(128),
                layers.LeakyReLU(alpha=0.2),
                # layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
                layers.Dense(self.zernike_dim),
            ],
            name="generator",
        )
        generator.summary()
        return generator


if __name__ == "__main__":
    ...
    # gan_helper = GANHelper()
    # train_histories = gan_helper.train(300)
    # d_loss = [item for h in train_histories for item in h.history["d_loss"]]
    # g_loss = [item for h in train_histories for item in h.history["g_loss"]]
    #
    # plt.plot(range(len(d_loss)), d_loss, label="d_loss")
    # plt.plot(range(len(g_loss)), g_loss, label="g_loss")
    # plt.legend()
    # plt.show()
    # gan_trainer.train(gan_trainer.load_data(), epochs=500)
    # coeffs: np.ndarray = gan_trainer.evaluate(1)
    # ps = PhaseScreen()
    # ps.set_zernike_coeffients(list(coeffs[0]))
    # plt.imshow(ps.get_screen())
    # ps.simulate(5)
    # plt.imshow(ps.get_screen())
    # input()
