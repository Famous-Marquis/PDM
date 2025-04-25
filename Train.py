import os
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from Diffusion import DDPM
from Model import FCMean, FCCov
from aberration import ZERNIKE_NUMS
from config import MODEL_CONFIG
from generate_data import generate_data
from plot_struct import plot_struct
import numpy as np
from scipy.linalg import sqrtm


def frechet_distance(x, y, epsilon=1e-6, verbose=True):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    def calculate_mean_cov(z):
        mean = np.mean(z, axis=0)
        cov = np.cov(z, rowvar=False)
        return mean, cov

    # Step 1: 均值和协方差
    mu_x, cov_x = calculate_mean_cov(x)
    mu_y, cov_y = calculate_mean_cov(y)

    # Step 2: 添加正则项以增强协方差稳定性
    cov_x += np.eye(cov_x.shape[0]) * epsilon
    cov_y += np.eye(cov_y.shape[0]) * epsilon

    # Step 3: 矩阵乘积开方计算
    cov_prod = cov_x @ cov_y
    sqrt_prod = sqrtm(cov_prod)

    # Step 4: 如果有复数误差，取实部 + 强制对称化
    if np.iscomplexobj(sqrt_prod):
        if verbose:
            print("Warning: sqrtm produced complex values. Taking real part.")
        sqrt_prod = np.real(sqrt_prod)

    sqrt_prod = (sqrt_prod + sqrt_prod.T) / 2.0  # 强制对称化（关键）

    # Step 5: 检查 sqrt_prod 是否异常
    if not np.all(np.isfinite(sqrt_prod)):
        raise ValueError("Matrix sqrtm failed: contains NaN or Inf")

    # Step 6: 计算 trace 和最终的 FID
    diff = mu_x - mu_y
    trace_term = np.trace(cov_x + cov_y - 2 * sqrt_prod)
    fid = diff.dot(diff) + trace_term

    # Step 7: Debug 输出
    if verbose:
        print("Mean squared difference:", diff.dot(diff))
        print("Trace term:", trace_term)
        print("FID (before clip):", fid)

    return max(fid, 0.0)


class DDPMMonitor(keras.callbacks.Callback):
    def __init__(self, real_data, model_checkpoint_path):
        self.real_data = real_data
        self.model: DDPM
        self.fds = []
        self.model_checkpoint_path = model_checkpoint_path

    def on_train_begin(self, logs=None):
        # 绘制正向扩散过程
        coeff = self.real_data[:10]
        coeff = tf.convert_to_tensor(coeff, dtype=tf.float32)
        self.model.show_diffusion(coeff)

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 == 0:
        max_len = min(4000, self.real_data.shape[0])

        x_T = tf.random.normal((max_len, self.real_data.shape[1]))
        generated_data = self.model.denoise(x_T)
        generated_data = generated_data.numpy()
        fd = frechet_distance(self.real_data[:max_len], generated_data)
        self.fds.append(fd)

        logs["val_loss"] = fd
        print("epoch:", epoch, " Frechet distance:", fd)

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
        Dphi_1d_mean_eval,r_over_r0_eval=plot_struct(x_0_eval)
        Dphi_1d_mean,r_over_r0=plot_struct(self.real_data[:maxlen])
        fig2 = plt.figure(figsize=(6,4))
        plt.plot(r_over_r0, Dphi_1d_mean, 'bo-', label='data $D_\phi(r)$')
        plt.plot(r_over_r0_eval, Dphi_1d_mean_eval, 'ro-', label='sampled $D_\phi(r)$')
        plt.xlabel(r'$r / r_0$')
        plt.ylabel(r'$D_\phi(r)$')
        plt.title('1D radial structure function')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./SampledImgs/DDPM_struct1.png", dpi=300)
        plt.close(fig2)
        # 随机绘制多个原始样本与生成样本
        fig,axes = plt.subplots(6,6,figsize=(10,5))
        for i,ax in enumerate(axes.flat):
            bar=ax.imshow(x_0_eval[i][None,:],aspect='auto',cmap='viridis')
            fig.colorbar(bar, ax=ax,orientation='vertical')
            ax.set_yticks([])
        fig.suptitle('DDPM generated samples')
        fig.tight_layout()
        plt.savefig("./SampledImgs/DDPM_samples1.png", dpi=300)
        plt.close(fig)

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
        return DDPM(model=model, model_v=model_v,
                    beta_1=self.model_config["beta_1"],
                    beta_T=self.model_config["beta_T"], T=self.model_config["T"],
                    cosine_schedule=self.model_config["cosine_schedule"], )

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
            loss_epoch = history.history['loss_simple']
            fig = plt.figure()
            plt.plot(loss_epoch)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()
            # plt.savefig("./Record/loss_epoch.png")
            plt.close(fig)

    def load_data(self, batch_size):
        if self.model_config["generate_new_data"]:
            generate_data(length_per_Dr0=self.model_config["length_per_Dr0"],
                          nums_Dr0=self.model_config["nums_Dr0"],
                          Dr0_range=self.model_config["Dr0_range"], )
        data_series = pandas.read_pickle(self.data_path)
        data_matrix = np.stack(data_series).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data_matrix)
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
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
            ]
        )

        self._plot_loss(history)

if __name__ == '__main__':
    trainer = DDPMTrainer(model_config=MODEL_CONFIG)
    tfdataset,datamatrix=trainer.load_data(64)
    D_phi,r_over_r0=plot_struct(datamatrix[:100])
    plt.plot(D_phi,r_over_r0)
    plt.show()


