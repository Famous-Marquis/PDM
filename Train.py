import os

import numpy as np
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from tensorflow import keras

from Diffusion import DDPM
from Model import FCMean, FCCov
from aberration import ZERNIKE_NUMS
from config import MODEL_CONFIG
from metrics import frechet_distance, compare_pca_spectrum, FD_calculator
from plot_struct import plot_struct, plot_struct_curve

class DDPMMonitor(keras.callbacks.Callback):
    def __init__(self, real_data, model_checkpoint_path):
        self.real_data = real_data
        self.model: DDPM
        self.fds = []
        self.pcas=[]
        self.model_checkpoint_path = model_checkpoint_path
        self.FD_calculator=FD_calculator(real_data)

    def on_train_begin(self, logs=None):
        # 绘制正向扩散过程
        coeff = self.real_data[:10]
        coeff = tf.convert_to_tensor(coeff, dtype=tf.float32)
        self.model.show_diffusion(coeff)

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 == 0:
        max_len = min(5000, self.real_data.shape[0])

        x_T = tf.random.normal((max_len, self.real_data.shape[1]))
        generated_data = self.model.denoise(x_T)
        generated_data = generated_data.numpy()
        fd = self.FD_calculator.frechet_distance(generated_data)
        self.fds.append(fd)
        # pca=compare_pca_spectrum(self.real_data[:max_len], generated_data)
        # self.pcas.append(pca)
        logs["val_loss"] = fd
        print("epoch:", epoch, " Frechet distance:",fd)

    def on_train_end(self, logs=None):
        fig1 = plt.figure()
        plt.plot(self.fds)
        plt.xlabel('epoch ')
        plt.ylabel("Frechet distance")
        plt.show()
        plt.close(fig1)
        maxlen = min(500, self.real_data.shape[0])
        #  绘制反向扩散过程
        x_T = tf.random.normal(shape=(maxlen, ZERNIKE_NUMS))
        x_0_eval = self.model.show_denoise(x_T)
        x_0_eval = x_0_eval.numpy()
        # 绘制结构函数
        Dphi_1d_mean_eval, Dphi_1d_std_eval, r_over_r0_eval = plot_struct(x_0_eval)
        Dphi_1d_mean, Dphi_1d_std, r_over_r0 = plot_struct(self.real_data[:maxlen])
        plot_struct_curve(r_over_r0, Dphi_1d_mean, Dphi_1d_std)
        plot_struct_curve(r_over_r0, Dphi_1d_mean_eval, Dphi_1d_std_eval)
        fig2 = plt.figure(figsize=(6, 4))
        plt.plot(r_over_r0, Dphi_1d_mean, 'bo-', label='data $D_\phi(r)$')
        plt.plot(r_over_r0_eval, Dphi_1d_mean_eval, 'ro-', label='sampled $D_\phi(r)$')
        plt.xlabel(r'$r / r_0$')
        plt.ylabel(r'$D_\phi(r)$')
        plt.title('1D radial structure function')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./SampledImgs/{self.model.model_name}_struct.png", dpi=300)
        plt.close(fig2)
        # 随机绘制多个原始样本与生成样本
        fig, axes = plt.subplots(3, 3, figsize=(10, 5))
        for i, ax in enumerate(axes.flat):
            bar = ax.imshow(x_0_eval[i][None, :], aspect='auto', cmap='viridis')
            fig.colorbar(bar, ax=ax, orientation='vertical')
            ax.set_yticks([])
        fig.suptitle(f'{self.model.model_name} generated samples')
        fig.tight_layout()
        plt.savefig(f"./SampledImgs/{self.model.model_name}_samples.png", dpi=300)
        plt.close(fig)


class DDPMTrainer:
    def __init__(self, model_config,data_path):
        self.model_name = model_config["model_name"]
        self.model_config = model_config
        self.data_path = data_path
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
        return DDPM(model=model, model_v=model_v,model_name=self.model_config['model_name'],
                    beta_1=self.model_config["beta_1"],
                    beta_T=self.model_config["beta_T"], T=self.model_config["T"],
                    cosine_schedule=self.model_config["cosine_schedule"],
                    predict_cov=self.model_config["predict_cov"])

    def _load_weights(self):
        if self.model_config["load_weights"]:
            if os.path.exists(self.model_checkpoint_path + ".index"):
                # if os.path.exists(self.model_checkpoint_path):
                try:
                    self.ddpm.load_weights(self.model_checkpoint_path)
                except Exception as e:
                    print("Weights load failed, {}".format(e))
                else:
                    print("Weights load succeeded")
            else:
                print("No checkpoint found")

    def _plot_loss(self, history):
        if self.model_config["predict_cov"]:
            ...
        # (协方差)todo: 完善 预测协方差的绘图
        else:
            loss_epoch = history.history['loss']
            fig = plt.figure()
            plt.plot(loss_epoch)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()
            # plt.savefig("./Record/loss_epoch.png")
            plt.close(fig)

    def load_data(self, batch_size):
        data_matrix = np.load(self.data_path).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data_matrix)
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
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
                DDPMMonitor(real_data=data_matrix,
                            model_checkpoint_path=self.model_checkpoint_path),
                keras.callbacks.ModelCheckpoint(self.model_checkpoint_path, save_weights_only=True,
                                                save_best_only=True),
            ],
            shuffle=True
        )

        self._plot_loss(history)


if __name__ == '__main__':
    trainer = DDPMTrainer(model_config=MODEL_CONFIG)
    tfdataset, datamatrix = trainer.load_data(64)
    D_phi, r_over_r0 = plot_struct(datamatrix[:100])
    plt.plot(D_phi, r_over_r0)
    plt.show()
