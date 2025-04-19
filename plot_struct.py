import numpy as np
from tqdm import tqdm
from aberration import PhaseScreen
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.signal import fft2d, fftshift, ifft2d, ifftshift

def radial_profile_2d_2(data, center=None, nbins=None):
    """
    对二维结构函数进行环形平均，得到一维结构函数。
    参数:
        data: 2D numpy array，相位结构函数
        center: 环平均的中心 (cx, cy)，默认为图像中心
        nbins: 半径的分 bin 数量，越大越精细
    返回:
        r_bin_centers: 每个 bin 的中心半径
        radial_mean: 每个 bin 对应的平均值
    """
    N = data.shape[0]
    if center is None:
        center = (N / 2, N / 2)
    cx, cy = center

    # 生成径向坐标
    y, x = np.indices(data.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # 设置 bin
    if nbins is None:
        nbins = N//10
    r_max = np.max(r)
    r_bins = np.linspace(0, r_max, nbins + 1)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # 使用 np.digitize 给每个像素分配一个 bin 索引
    r_flat = r.ravel()
    data_flat = data.ravel()
    bin_indices = np.digitize(r_flat, r_bins) - 1  # digitize返回的是1-based索引

    # 避免超范围索引
    valid = (bin_indices >= 0) & (bin_indices < nbins)

    radial_sum = np.bincount(bin_indices[valid], weights=data_flat[valid], minlength=nbins)
    radial_count = np.bincount(bin_indices[valid], minlength=nbins)

    with np.errstate(divide='ignore', invalid='ignore'):
        radial_mean = np.true_divide(radial_sum, radial_count)
        radial_mean[radial_count == 0] = 0  # 避免除以0

    return r_bin_centers, radial_mean
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
        phi = scr.get_screen()
        phi = tf.cast(tf.convert_to_tensor(phi), tf.complex128)
        mask = tf.cast(tf.convert_to_tensor(np.ones([SIZE, SIZE])), tf.complex128)
        strn = abs(str_fcn2_ft(phi, mask, 1)).numpy()
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