import os
import time
from typing import Union
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.callbacks
from keras import layers
import tensorflow.python.keras.backend as K
import keras
import keras.losses
from tensorflow.keras.optimizers import Adam
import sys

from Main import MODEL_CONFIG

print(sys.path)

# from sklearn.model_selection import train_test_split
from aberration import PhaseScreen, ZERNIKE_NUMS
import random
from tqdm import tqdm
import numpy as np
import warnings

"""
# Jensen-Shannon Divergence
"""


def calculate_mean_cov(x):
    """
    ## 计算给定数据集的均值和协方差矩阵。

    parameters:
    ---------------
    x: numpy.ndarray 输入数据，形状为(N, M)，其中N是样本数，M是特征数。

    返回值:
    ---------------
    包含两个元素的元组，第一个元素是均值向量，第二个元素是协方差矩阵。

    说明:
    - 首先计算数据集的大小，即样本数。
    - 初始化均值向量`mean`和协方差矩阵`cov`，它们的大小与输入数据的特征数相同。
    - 使用numpy的sum函数沿着轴0（即样本轴）计算均值。
    - 计算协方差矩阵，通过转置输入数据矩阵`x`，然后与自身点乘，最后除以样本数。
    """

    size = x.shape[0]
    mean = np.zeros(size)
    cov = np.zeros((size, size))
    N = x.shape[0]

    mean = x.sum(axis=0) / N
    # ？？点积与矩阵乘法？？
    cov = x.T.dot(x) / N
    return mean, cov


def frechet_distance(mean1, cov1, mean2, cov2):
    """
    计算两个高斯分布之间的Frechet距离（也称为Wasserstein-2距离）。

    参数:
    mean1, mean2 : numpy.ndarray
        两个高斯分布的均值向量。
    cov1, cov2 : numpy.ndarray
        两个高斯分布的协方差矩阵。

    返回:
    float
        两个高斯分布之间的Frechet距离。

    说明:
    - Frechet距离是衡量两个概率分布差异的一种方法，特别适用于高斯分布。
    - 首先，检查协方差矩阵是否有非正的特征值，如果有，则警告用户，因为它们可能导致距离计算不准确。
    - 使用numpy的linalg.eigh函数计算协方差矩阵的特征值和特征向量，并确保特征值非负。
    - 计算协方差矩阵的平方根，然后计算两个协方差矩阵的乘积。
    - 计算乘积矩阵的特征值，再次确保特征值非负。
    - 计算Frechet距离的公式，包括均值向量差的平方和，以及协方差矩阵特征值的和。
    """

    def check_nonpositive_eigvals(l):
        nonpos = l < 0
        if nonpos.any():
            warnings.warn(
                "Rank deficient covariance matrix, "
                "Frechet distance will not be accurate.",
                Warning,
            )
        l[nonpos] = 0

    (l1, v1) = np.linalg.eigh(cov1)
    check_nonpositive_eigvals(l1)
    cov1_sqrt = (v1 * np.sqrt(l1)).dot(v1.T)
    cov_prod = cov1_sqrt.dot(cov2).dot(cov1_sqrt)
    lp = np.linalg.eigvalsh(cov_prod)
    check_nonpositive_eigvals(lp)

    trace = l1.sum() + np.trace(cov2) - 2 * np.sqrt(lp).sum()
    diff_mean = mean1 - mean2
    fd = diff_mean.dot(diff_mean) + trace

    return fd


def FD(y_pred, y_true):

    mean1, cov1 = calculate_mean_cov(y_true)
    mean2, cov2 = calculate_mean_cov(y_pred)
    fd = frechet_distance(mean1, cov1, mean2, cov2)
    return fd


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
    g_optimizer : keras.optimizers.Optimizer
        生成器模型的优化器。
    loss_fn : keras.losses.Loss
        用于训练GAN的损失函数。

    方法:
    ------------
    1.compile : 编译模型，设置优化器和损失函数。
    2.train_step : 进行单步训练，更新判别器和生成器的权重。

    """

    def __init__(self, discriminator, generator, latent_dim, test_set):

        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.test_set = test_set

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

        # 返回判别器和生成器的损失
        return {"d_loss": d_loss, "g_loss": g_loss}


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
        self.fds_exist = False

    def on_epoch_end(self, epoch, logs=None):

        # time.sleep(0.2)
        random_latent_vectors = tf.random.normal(
            shape=(self.real_samples_num, self.latent_dim)
        )
        generated_zernike = self.model.generator(random_latent_vectors)
        fd = FD(generated_zernike.numpy(), self.real_zernike)
        print("epoch {} FD : {}".format(epoch, fd))
        # print("#### 调用 #####")
        if not self.fds_exist:
            self.fds = np.array(fd)
            self.fds_exist = True
        else:
            if fd < self.fds.min():
                self.fds = np.append(self.fds, fd)
                self.model.save_weights("./tmp/checkpoint_gan/")


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

    def __init__(self):
        self.zernike_dim = 64
        self.latent_dim = 128
        self.data_length = 4000
        self.model_config = MODEL_CONFIG
        self.ckpt_path = self.model_config["gan_checkpoint_path"]
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        self.scr = PhaseScreen()
        self.gan = GAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=self.latent_dim,
            test_set=self.load_data(),
        )

        self.gan.compile(
            d_optimizer=Adam(learning_rate=0.0005),
            g_optimizer=Adam(learning_rate=0.0005),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        )

    def train(self, x, epochs=100, restore=True):
        dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32)))
        dataset = dataset.shuffle(buffer_size=1024).batch(400).prefetch(64)

        checkpoint_dir = "../gan/tmp/checkpoint_gan/"
        try:
            self.gan.load_weights(checkpoint_dir)
        except:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            print("load weights fail")

        callbacks = []
        # callbacks.append(
        #     tf.keras.callbacks.ModelCheckpoint(
        #         checkpoint_dir,
        #         monitor="jsd",
        #         save_best_only=True,
        #         save_weights_only=True,
        #         mode="min",
        #     )
        # )
        callbacks.append(GANMonitor(self.load_data()))

        history = self.gan.fit(
            dataset,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def repeat_train(self, repeat_num=10, data_update_iter=3):
        train_histories = []
        for i in range(repeat_num):
            if i % data_update_iter == 0:
                x = self.update_data()
            history = self.train(x)
            train_histories.append(history)
        return train_histories

    def load_data(
        self,
    ):
        # prepare test data
        data_path = "../data/gan/gan.npy"
        return np.load(data_path, allow_pickle=True)

    def update_data(
        self,
    ):
        # todo: 关闭这个功能
        # Dr0 = [1, 3, 5, 7, 9, 11, 13, 15]
        zs = []
        with tqdm(total=self.data_length) as pbar:
            for _ in range(self.data_length):
                dr0 = random.uniform(5, 5.9)
                self.scr.simulate_turbulence(Dr0=dr0, method="zernike")
                z = self.scr.get_coeffients()[:ZERNIKE_NUMS]
                zs.append(z)
                pbar.update(1)
        return np.array(zs)

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
                layers.Dense(64, activation="tanh"),

            ],
            name="generator",
        )
        generator.summary()
        return generator

    def evaluate(self, num=3):
        z = tf.random.normal(shape=(num, self.latent_dim))
        self.gan.load_weights("../gan/tmp/checkpoint_gan/")
        gen_zernike = self.gan.generator(z)
        return gen_zernike.numpy()


if __name__ == "__main__":

    gan_helper = GANHelper()
    train_histories = gan_helper.repeat_train(repeat_num=20)
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
