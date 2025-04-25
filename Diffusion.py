import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import losses, optimizers

from aberration import ZERNIKE_NUMS


def extract(v, t, x_shape):
    """
    提取 v 中的值，并广播到目标形状 x_shape。

    参数:
        v: 一维张量，包含系数序列。
        t: 索引值（标量或张量）。
        x_shape: 目标形状。
    """
    # 提取值，注意 t-1 是索引
    t = tf.cast(t, tf.int32)
    out = tf.gather(v, t - 1, axis=0)
    out = tf.cast(out, tf.float32)  # 转换为浮点类型
    while len(out.shape) < len(x_shape):
        out = tf.expand_dims(out, -1)
    # 直接广播到目标形状
    out = tf.broadcast_to(out, x_shape)  # 广播到目标形状

    return out


class DDPM(keras.Model):
    def __init__(self, model: keras.Model, beta_1, beta_T, T, cosine_schedule=False,
                 predict_cov=False, model_v=None, ):
        super(DDPM, self).__init__()
        self.L_t_record = None
        self.optimizer = None
        self.cov_optimizer = None
        self.loss_fn = None
        self.model = model
        self.model_v = model_v
        self.T = T
        self.cosine_schedule = cosine_schedule
        self.predict_cov = predict_cov

        self.loss_tracker = keras.metrics.Mean(name='loss')
        if self.cosine_schedule:
            # 余弦调度β
            """一般来说，系数总体呈现统一的趋势。若发现某个单调性变化之处或“inf”“nan”,很有可能出错
            """
            t = tf.range(1, T + 1 + 5)
            s = 1 / ZERNIKE_NUMS
            f_t = tf.square(tf.math.cos((t / (self.T + 5) + s) * np.pi / 2 / (1 + s)))
            f_t = tf.clip_by_value(f_t, 1e-5, 1 - 1e-5)
            self.alpha_bar = tf.clip_by_value(f_t / f_t[0] + 1e-5, 1e-5, 1 - 1e-5)[:-5]
            self.alpha_bar_prev = tf.concat([[1.], self.alpha_bar[:-1]], axis=0)
            assert self.alpha_bar.shape[0] == self.T
            self.alpha = self.alpha_bar / self.alpha_bar_prev
            self.beta = 1 - self.alpha
            self.beta = tf.clip_by_value(self.beta, 1e-5, 1 - 1e-5)
            self.beta_tilde = (1 - self.alpha_bar_prev) / (1 - self.alpha_bar) * self.beta
            self.one_div_sqrt_alpha = 1. / tf.math.sqrt(self.alpha)
            self.sqrt_alpha_bar = tf.math.sqrt(self.alpha_bar)
            self.one_minus_alpha_bar = (1 - self.alpha_bar)
            self.sqrt_one_minus_alpha_bar = tf.math.sqrt(1. - self.alpha_bar)
            self.beta_div_sqrt_one_minus_alpha_bar = self.beta / self.sqrt_one_minus_alpha_bar

            self.Lambda = 0.001
        else:
            # 线性调度β
            self.beta = tf.linspace(beta_1, beta_T, T)
            self.alpha = 1 - self.beta
            self.alpha_bar = tf.math.cumprod(self.alpha)
            self.alpha_bar = tf.clip_by_value(self.alpha_bar, 1e-4, 1 - 1e-4)
            self.alpha_bar_prev = tf.concat([[1.], self.alpha_bar[:-1]], axis=0)
            self.beta_tilde = (1 - self.alpha_bar_prev) / (1 - self.alpha_bar) * self.beta
            self.one_div_sqrt_alpha = 1. / tf.math.sqrt(self.alpha)
            self.sqrt_alpha_bar = tf.math.sqrt(self.alpha_bar)
            self.one_minus_alpha_bar = (1 - self.alpha_bar)
            self.sqrt_one_minus_alpha_bar = tf.math.sqrt(1. - self.alpha_bar)
            self.beta_div_sqrt_one_minus_alpha_bar = self.beta / self.sqrt_one_minus_alpha_bar

    def compile(self, lr_min, lr_max, warmup_steps, total_steps):
        # todo: 可能需要的学习率预热
        lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=lr_max,
                                                       alpha=lr_min,
                                                       decay_steps=total_steps)

        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = losses.MeanSquaredError()
        super(DDPM, self).compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=self.loss_fn)

        # 协方差
        if self.predict_cov:
            self.cov_optimizer = keras.optimizers.Adam(lr=lr_schedule)

            # self.L_t_record = np.zeros([self.T, 10], dtype=float)

        else:
            self.cov_optimizer = None

    def calculate_L_t(self, t, x_t, x_t_prev, mean_pred, cov_pred):
        k = x_t_prev.shape[1]
        n = x_t_prev.shape[0]
        if t == 0:
            log_likelihood = -0.5 * k * n * np.log(2 * np.pi) - 0.5 * n * \
                             np.linalg.slogdet(cov_pred)[1] - 0.5 * np.sum(
                (x_t_prev - mean_pred) @ np.linalg.inv(cov_pred) * (cov_pred - mean_pred))
            # (协方差)todo：完善Lt计算以及Lvlb
            ...
        elif t == self.T:
            ...
        else:
            ...

    def call(self, x_0):
        t = tf.random.uniform(shape=(x_0.shape[0],), minval=1, maxval=self.T,
                              dtype=tf.int32)

        # t: int32, 外部生成，每批t保持一致
        # t = tf.random.uniform(minval=1, maxval=self.T, shape=x_0.shape[0], dtype=tf.int32)
        eps = tf.random.normal(shape=x_0.shape, dtype=tf.float32)
        coeff1 = extract(self.sqrt_alpha_bar, t, x_0.shape)
        coeff2 = extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)
        x_t = coeff1 * x_0 + coeff2 * eps

        if self.predict_cov:
            eps_pred, v_pred = self.model(x_t, t), self.model_v(x_t, t)
        else:
            eps_pred, v_pred = self.model(x_t, t), None
        return x_t, eps, t, eps_pred, v_pred

    @tf.function
    def train_step(self, x_0):
        if self.predict_cov:
            with tf.GradientTape() as tape:
                x_t, eps, t, eps_pred, v_pred = self.call(x_0)
                # loss = L_hybrid
                # (协方差)todo: 将下列重复的代码封装到call()?

                param1 = extract(self.beta, t, x_t.shape)
                param2 = extract(self.beta_tilde, t, x_t.shape)
                cov_pred = tf.math.exp(
                    v_pred * tf.math.log(param1) + (1 - v_pred) * tf.math.log(param2))
                # (协方差)todo：后续补充此梯度下降
                ...
            ...

            # return x_t, eps, mean_pred, cov_pred

        # 非余弦调度，3个返回值 + None
        else:
            with tf.GradientTape() as eps_tape:
                x_t, eps, t, eps_pred, v_pred = self.call(x_0)
                loss_simple = self.loss_fn(eps_pred, eps)
            eps_grad = eps_tape.gradient(loss_simple, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(eps_grad, self.model.trainable_variables))

            self.loss_tracker.update_state(loss_simple)
            return {'loss_simple': loss_simple}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def show_diffusion(self, z_coeff):
        # z_coeff : (ZERNIKE_NUMS,) tf.Tensor
        # assert type(z_coeff) == tf.Tensor,"z_coeff must be of type tf.Tensor"

        sampled_t = np.linspace(0, self.T, 9).astype(int)
        z_coeff = z_coeff[0]
        z_coeff = z_coeff[None, :]
        fig, axes = plt.subplots(3, 3, figsize=(10, 5))
        axes = axes.flatten()
        for i, t in enumerate(sampled_t):
            if t == 0:
                z_coeff_t = z_coeff.numpy()
            else:
                eps = tf.random.normal(shape=z_coeff.shape, dtype=tf.float32)
                coeff1 = extract(self.sqrt_alpha_bar, t, z_coeff.shape)
                coeff2 = extract(self.one_minus_alpha_bar, t, z_coeff.shape)
                z_coeff_t = coeff1 * z_coeff + coeff2 * eps
                z_coeff_t = z_coeff_t.numpy()
            # z_coeff_t=z_coeff_t[None,:]
            bar = axes[i].imshow(z_coeff_t, aspect='auto', cmap=plt.get_cmap('viridis'))
            fig.colorbar(bar, ax=axes[i], orientation='vertical')
            axes[i].set_yticks([])
            axes[i].set_title("t = {}".format(t))

        fig.suptitle("Diffusion process")
        fig.tight_layout()
        plt.savefig("./SampledImgs/Diffusion_process1.png", dpi=300)
        plt.close(fig)
        print("Diffusion process plot successfully saved")

    @tf.function
    def denoise_step_cov(self, x_t, t):
        coeff1 = extract(self.one_div_sqrt_alpha, t, x_t.shape)
        coeff2 = extract(self.beta_div_sqrt_one_minus_alpha_bar, t, x_t.shape)

        if t > 1:
            z = tf.random.normal(shape=x_t.shape, dtype=tf.float32)
            v_pred = self.model_v(x_t, t)
            param1 = extract(self.beta, t, x_t.shape)
            param2 = extract(self.beta_tilde, t, x_t.shape)
            cov_t_pred = tf.math.exp(
                v_pred * tf.math.log(param1) + (1 - v_pred) * tf.math.log(param2))
            L = tf.linalg.cholesky(cov_t_pred)
            eps_pred = self.model(x_t, t)
            x_t_prev = coeff1 * (x_t - coeff2 * eps_pred) + tf.matmul(L, z)
        else:
            eps_pred = self.model(x_t, t)
            x_t_prev = coeff1 * (x_t - coeff2 * eps_pred)
        x_t = x_t_prev
        return x_t

    @tf.function
    def denoise_step(self, x_t, t):
        coeff1 = extract(self.one_div_sqrt_alpha, t, x_t.shape)
        coeff2 = extract(self.beta_div_sqrt_one_minus_alpha_bar, t, x_t.shape)
        # todo: beta or beta_tilde
        coeff3 = extract(self.beta, t, x_t.shape)
        eps_pred = self.model(x_t, t)

        # 使用 tf.cond 替代 if-else
        # if t.numpy()==1:
        #     print(t.numpy())
        x_t_prev = tf.cond(
            t > 1,
            lambda: coeff1 * (x_t - coeff2 * eps_pred) + coeff3 * tf.random.normal(shape=x_t.shape,
                                                                                   dtype=tf.float32),
            lambda: coeff1 * (x_t - coeff2 * eps_pred)
        )
        # print("step:",t.numpy())
        # print(x_t_prev[0].numpy())
        return x_t_prev

    @tf.function
    def denoise(self, x_T):
        x_t = x_T
        if self.predict_cov:
            for t in tf.range(self.T, 0, -1):
                x_t = self.denoise_step_cov(x_t, t)
            return x_t
        else:
            for t in tf.range(self.T, 0, -1):
                x_t = self.denoise_step(x_t, t)
            return x_t

    def show_denoise(self, x_T):
        x_t = x_T

        sampled_t = tf.constant(np.linspace(1, self.T, 9).astype(int))
        fig, axes = plt.subplots(3, 3, figsize=(10, 5))
        i = 0
        axes = axes.flatten()
        for t in tf.range(self.T, 0, -1):

            if self.predict_cov:
                x_t = self.denoise_step_cov(x_t, t)
            else:
                x_t = self.denoise_step(x_t, t)

            # 判断 t 是否在 sampled_t 中
            if tf.reduce_any(tf.equal(t, sampled_t)):
                coeff = x_t[0].numpy()
                coeff = coeff[None, :]
                bar = axes[i].imshow(coeff, aspect='auto', cmap=plt.get_cmap("viridis"))
                fig.colorbar(bar, ax=axes[i], orientation='vertical')
                axes[i].set_yticks([])
                axes[i].set_title("t = {}".format(t))
                i += 1
                if i == len(sampled_t):
                    fig.suptitle("Denoise process")
                    fig.tight_layout()
                    plt.savefig("./SampledImgs/Denoise_process1.png", dpi=300)
                    plt.close(fig)
                    print("Denoise process plot successfully saved")
        return x_t

    @tf.function
    def generate_samples(self, sample_num):
        batch_size = 100
        latent_dim = ZERNIKE_NUMS
        num_steps = sample_num // batch_size + 1
        sample_num = num_steps * batch_size

        # 创建一个 TensorArray 来存储生成的样本
        samples = tf.TensorArray(dtype=tf.float32, size=num_steps, dynamic_size=False)
        for step in range(num_steps):
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            generated_samples = self.denoise(noise)
            samples = samples.write(step, generated_samples)

        # 将 TensorArray 转换为一个张量
        samples = samples.stack()
        samples = tf.reshape(samples, (sample_num, -1))
        return samples


