import numpy as np
from tqdm import tqdm
from aberration import PhaseScreen
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.signal import fft2d, fftshift, ifft2d, ifftshift

def radial_profile_2d_2(data, center=None):
    N = data.shape[0]
    if center is None:
        center = (N / 2, N / 2)
    cx, cy = center

    # 生成径向坐标
    x, y = np.indices((N, N))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # 找到唯一的r值，并排序
    r_idx = np.unique(r)

    # 计算径向平均值
    radial_mean = []
    for item in r_idx:
        mask = (r == item)
        mean_value = np.mean(data[mask])
        radial_mean.append(mean_value)

    return r_idx, np.array(radial_mean)
def ft2(x, delta):
    return fftshift(fft2d(fftshift(x))) * delta ** 2
def ift2(x, delta_f):
    N = x.shape[0]
    return ifftshift(ifft2d(ifftshift(x))) * (N * delta_f) ** 2

def str_fcn2_ft(ph, mask, delta):
    N = ph.shape[0]
    ph = ph * mask
    P = ft2(ph, delta)
    S = ft2(ph ** 2, delta)
    W = ft2(mask, delta)

    delta_f = 1 / (N * delta)
    w2 = ift2(W * tf.math.conj(W), delta_f)
    sw = tf.cast(tf.math.real(S * tf.math.conj(W)) - tf.abs(P) ** 2, tf.complex128)
    D = (2 * ift2(sw, delta_f))
    # / w2 * mask)
    return D

def plot_struct(z_coes_array):
    SIZE = 256
    scr = PhaseScreen(N=SIZE)

    for i, z_coes in tqdm(enumerate(z_coes_array), total=z_coes_array.shape[0],desc="computing struct..."):
        scr.set_zernike_coeffients(list(z_coes))
        scr.add_pupil()
        psf = abs(scr.get_psf())
        psf = tf.cast(tf.convert_to_tensor(psf), tf.complex128)
        mask = tf.cast(tf.convert_to_tensor(np.ones([SIZE, SIZE])), tf.complex128)
        strn = abs(str_fcn2_ft(psf, mask, 1)).numpy()
        # plt.imshow(strn)
        # ================================
        # 1) 假设已有 2D 相位结构函数 Dphi_2d
        #    这里随机生成一个示例 (只作演示)
        Dphi_2d = strn  # 你实际应使用真正的结构函数数据

        # 2) 对 2D 数组做环平均
        radii_pix, Dphi_1d = radial_profile_2d_2(Dphi_2d)
        if i == 0:
            Dphi_1ds = np.zeros([z_coes_array.shape[0], Dphi_1d.shape[0]])
        Dphi_1ds[i] = Dphi_1d

    Dphi_1d_mean = np.mean(Dphi_1ds, axis=0)
    assert Dphi_1d_mean.shape == Dphi_1d.shape, "形状有误"
    # 3) 转换为 r/r0
    #    假设 1个像素 = 1个长度单位(可根据实际需求改成: r_phys = radii_pix * pixel_scale)
    r0 = 25  # 示例: 设定 Fried 参数 r0 = 10 (与像素同单位)
    r_over_r0 = radii_pix / r0
    return Dphi_1d_mean,r_over_r0

    # 4) 绘图

    # plt.figure(figsize=(6, 4))
    # plt.plot(r_over_r0_generated, Dphi_1d_mean_generated, 'bo-', label='Radial $D_\phi(r)$')
    #
    # plt.xlabel(r'$r / r_0$')
    # plt.ylabel(r'$D_\phi(r)$')
    # plt.title('1D radial structure function from DDPM')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
if __name__ == "__main__":
    r0 = 0.2  # 示例: Fried 参数
    r_max = 1.0  # 例如你的口径大小或更大范围

    # 1) 生成无量纲坐标
    x_theo = np.linspace(0, r_max / r0, 200)  # 0到(r_max/r0)均匀200点

    # 2) 计算Kolmogorov理论
    D_theo = 6.88 * (x_theo ** (5.0 / 3.0))

    # 3) 绘制
    plt.figure(figsize=(6, 4))
    plt.plot(x_theo, D_theo, 'r--', label='Kolmogorov theory')
    plt.xlabel(r'$r / r_0$')
    plt.ylabel(r'$D_\phi(r)$')
    plt.title('Kolmogorov $D_\phi(r)$ vs. $r/r_0$')
    plt.legend()
    plt.grid(True)

    # plt.plot(r_over_r0, Dphi_1d_mean, 'bo-', label='Radial Dphi(r)')
    # plt.xlabel(r'$r / r_0$')
    # plt.ylabel(r'$D_\phi(r)$')
    # plt.title('1D radial structure function from 2D data')
    # plt.grid(True)
    # plt.legend()
    plt.savefig("./SampledImgs/Kolmogorov.png")