import os
import warnings

import numpy as np
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from Diffusion import GaussianDiffusion
from Model import FCMean, FCCov
from aberration import ZERNIKE_NUMS


def Frechet_distance(x, y):
    def calculate_mean_cov(z):
        # l: [N, ZERNIKE_NUMS]
        assert type(z) is np.ndarray

        mean = np.mean(z, axis=0)
        cov = np.cov(z, rowvar=False)
        return mean, cov

    def check_nonsensitive_eigenvalues(l):
        nonpos = l < 0
        if np.any(l):
            warnings.warn("Non-positive eigenvalues, FID may not work")
            l[nonpos] = 0
        return l

    assert type(x) is np.ndarray
    assert type(y) is np.ndarray

    mean_x, cov_x = calculate_mean_cov(x)
    mean_y, cov_y = calculate_mean_cov(y)
    l1, v1 = np.linalg.eigh(cov_x)
    l2, v2 = np.linalg.eigh(cov_y)
    l1 = check_nonsensitive_eigenvalues(l1)
    l2 = check_nonsensitive_eigenvalues(l2)
    sqrt_cov_x = v1 @ np.diag(np.sqrt(l1)) @ v1.T
    sqrt_cov_y = v2 @ np.diag(np.sqrt(l2)) @ v2.T

    cov_prod = sqrt_cov_x @ cov_y @ sqrt_cov_y
    l_prod, vec_prod = np.linalg.eigh(cov_prod)
    l_prod = check_nonsensitive_eigenvalues(l_prod)
    sqrt_prod = vec_prod @ np.diag(np.sqrt(l_prod)) @ vec_prod.T
    trace = np.trace(cov_x + cov_y - 2 * sqrt_prod)

    diff_mean = mean_x - mean_y
    FD = diff_mean.dot(diff_mean) + trace
    return FD


class DDPMMonitor(keras.callbacks.Callback):
    def __init__(self, real_data):

        self.real_data = real_data
        self.model: GaussianDiffusion
        self.fds = []

    def on_train_begin(self, logs=None):
        # 绘制正向扩散过程
        coeff = self.real_data[:10]
        coeff = tf.convert_to_tensor(coeff, dtype=tf.float32)
        self.model.show_diffusion(coeff)

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 == 0:
        max_len = min(500, self.real_data.shape[0])

        x_T = tf.random.normal((max_len, self.real_data.shape[1]))
        generated_data = self.model.denoise(x_T)
        generated_data = generated_data.numpy()
        fd = Frechet_distance(self.real_data[:max_len], generated_data)
        self.fds.append(fd)
        logs["val_loss"] = fd
        print("epoch:", epoch, " Frechet distance:", fd)

    def on_train_end(self, logs=None):
        fig1 = plt.figure()
        plt.plot(self.fds)
        plt.xlabel('epoch ')
        plt.ylabel("Frechet distance")
        plt.show()
        plt.savefig("./Record/FD-epoch.png")
        #  绘制反向扩散过程
        x_T = tf.random.normal(shape=(100, ZERNIKE_NUMS))
        x_0 = self.model.show_denoise(x_T)

        # todo：绘制结构函数，并选择指标，计算二者相似度


class DDPMTrainer:

    def __init__(self, model_config):
        self.model_config = model_config
        self.data_path = model_config["data_path"]
        self.model_checkpoint_path = model_config["model_checkpoint_path"]
        # self.model_v_checkpoint_path = model_config["model_v_checkpoint_path"]
        self.ddpm = self._build_ddpm()

    def _build_model(self):
        model = FCMean(d_model=self.model_config["d_model"],
                       model_struct=self.model_config["model_mean_struct"])
        if self.model_config["predict_cov"]:
            model_v = FCCov(d_model=self.model_config["d_model"],
                            model_struct=self.model_config["model_v_struct"])
        else:
            model_v = None
        return model, model_v

    def _build_ddpm(self):
        model, model_v = self._build_model()
        return GaussianDiffusion(model=model, model_v=model_v,
                                 beta_1=self.model_config["beta_1"],
                                 beta_T=self.model_config["beta_T"], T=self.model_config["T"],
                                 cosine_schedule=self.model_config["cosine_schedule"], )

    def _load_weights(self):
        # todo: 模型权重加载失败，需要看官网指南
        if self.model_config["load_weights"]:
            if os.path.exists(self.model_checkpoint_path + ".index"):
                try:
                    self.ddpm.model.load_weights(self.model_checkpoint_path)
                except Exception as e:
                    print("Weights load failed, {}".format(e))
                else:
                    print("Weights load succeeded")
            else:
                print("No checkpoint found")

    def _plot_loss(self, history):
        if self.model_config["predict_cov"]:
            ...
        # todo: finish predict plot
        else:
            loss_epoch = history.history['loss_simple']
            fig = plt.figure()
            plt.plot(loss_epoch)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            # plt.show()
            plt.savefig("./Record/loss_epoch.png")
            plt.close(fig)

    def load_data(self, batch_size):
        data_series = pandas.read_pickle(self.data_path)
        data_matrix = np.stack(data_series).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data_matrix)
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset, data_matrix

    def summary(self):
        self.ddpm.model.summary()
        if self.model_config["predict_cov"]:
            self.ddpm.model_v.summary()
        self.ddpm.summary()

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.model_config["epochs"]
        batch_size = self.model_config["batch_size"]
        # 加载数据
        tf_dataset, data_matrix = self.load_data(batch_size)
        # 编译模型
        total_steps = len(tf_dataset) // batch_size
        warmup_steps = int(total_steps * 0.1)
        self.ddpm.compile(self.model_config["lr_min"], self.model_config["lr_max"],
                          total_steps=total_steps, warmup_steps=warmup_steps)

        self._load_weights()

        history = self.ddpm.fit(
            tf_dataset,
            epochs=epochs,
            callbacks=[
                DDPMMonitor(real_data=data_matrix),
                keras.callbacks.ModelCheckpoint(
                    filepath=self.model_checkpoint_path,
                    save_best_only=True,
                    save_weights_only=True
                )
            ]
        )

        self._plot_loss(history)
