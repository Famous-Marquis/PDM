import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from aberration import PhaseScreen, BatchPhaseScreen, SCREEN_SIZE


def radial_profile_2d_batch(batch_data, r, center=None, nbins=None):
    """
    对 batch 的二维结构函数进行环形平均，得到 batch 的一维结构函数。
    参数:
        batch_data: [batch, N, N] numpy数组
        r: [N, N] numpy数组，表示每个像素到中心的归一化半径
        center: (cx, cy)，默认图像中心
        nbins: 半径分段数
    返回:
        r_bin_centers: [nbins] 每个 bin 的中心半径
        radial_mean: [batch, nbins] 每个batch、每个半径bin的平均值
    """

    batch, N, _ = batch_data.shape

    if center is None:
        center = (N / 2, N / 2)
    cx, cy = center

    mask = r < 1  # 只在 r<1 范围内做环形平均

    if nbins is None:
        nbins = N // 5

    r_max = np.max(r[mask])
    r_bins = np.linspace(0, r_max, nbins + 1)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # 将 r 展平，准备给所有 pixel 分配 bin
    r_flat = r.ravel()  # [N*N]
    bin_indices = np.digitize(r_flat, r_bins) - 1  # [N*N]，1-based转成0-based

    # 只保留有效区域
    valid = (bin_indices >= 0) & (bin_indices < nbins) & (r_flat < r_max)
    bin_indices_valid = bin_indices[valid]  # [num_valid_pixels]
    num_valid_pixels = bin_indices_valid.shape[0]

    # 展平 batch_data，每张图展成 [batch, N*N]
    data_flat = batch_data.reshape(batch, -1)  # [batch, N*N]
    data_valid = data_flat[:, valid]  # [batch, num_valid_pixels]

    # 初始化累加器
    radial_sum = np.zeros((batch, nbins), dtype=batch_data.dtype)
    radial_count = np.bincount(bin_indices_valid, minlength=nbins)

    # 对每个 bin，用广播加权累加
    for i in range(nbins):
        mask_bin = (bin_indices_valid == i)  # [num_valid_pixels]
        if np.any(mask_bin):
            radial_sum[:, i] = np.sum(data_valid[:, mask_bin], axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        radial_mean = np.true_divide(radial_sum, radial_count)
        radial_mean[:, radial_count == 0] = 0

    return r_bin_centers, radial_mean


def radial_profile_2d_2(batch_data, r, center=None, nbins=None):
    """
    对二维结构函数进行环形平均，得到一维结构函数。
    参数:
        batch_data: [batch,N,N] 3D numpy array，相位结构函数
        center: 环平均的中心 (cx, cy)，默认为图像中心
        nbins: 半径的分 bin 数量，越大越精细
    返回:
        r_bin_centers: 每个 bin 的中心半径
        radial_mean: 每个 bin 对应的平均值
    """
    batch = batch_data.shape[0]
    N = batch_data.shape[1]
    if center is None:
        center = (N / 2, N / 2)
    cx, cy = center

    # 只对r<1做环平均
    mask = r < 1
    # 设置 bin
    if nbins is None:
        nbins = N // 5
    r_max = np.max(r[mask])
    r_bins = np.linspace(0, r_max, nbins + 1)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # 使用 np.digitize 给每个像素分配一个 bin 索引
    r_flat = r.ravel()
    data_flat = batch_data.ravel()
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
    """
    x: [batch, N, N] 或 [N, N]，需要是复数或实数
    delta: 像素间隔（float标量）
    """
    x = tf.cast(x, tf.complex128)
    delta = tf.cast(delta, tf.complex128)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))),
                              axes=(-2, -1)) * delta ** 2


def ift2(x, delta_f):
    """
    2D反傅里叶变换，并正确缩放
    x: 输入频域数据
    delta_f: 频率间隔
    """
    N = tf.shape(x)[-1]  # 假设最后两维是 N x N
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))),
                               axes=(-2, -1)) \
        * (tf.cast(N, tf.complex128) ** 2) * (tf.cast(delta_f, tf.complex128) ** 2)


def str_fcn2_ft(ph, mask, delta):
    """
    ph:[N,N] 2D numpy array
    mask:[N,N] 2D numpy array(圆掩膜)
    """
    N = ph.shape[0]
    ph = ph * mask
    P = ft2(ph, delta)
    S = ft2(ph ** 2, delta)
    W = ft2(mask, delta)

    delta_f = 1 / (N * delta)
    w2 = ift2(W * tf.math.conj(W), delta_f)
    sw = tf.cast(tf.math.real(S * tf.math.conj(W)) - tf.abs(P) ** 2, tf.complex128)
    D = (2 * ift2(sw, delta_f)) / w2 * mask
    return D


def str_fcn2_ft_batch(ph, mask, delta):
    """
    批次版结构函数计算
    参数:
        ph: [batch, N, N] 或 [N, N]，相位屏
        mask: [batch, N, N] 或 [N, N]，掩膜
        delta: 像素间隔
    返回:
        D: [batch, N, N] 结构函数
    """
    # 自动扩展维度以支持无batch输入
    if len(ph.shape) == 2:
        ph = ph[None, ...]  # [1, N, N]
    if len(mask.shape) == 2:
        mask = mask[None, ...]  # [1, N, N]

    ph = tf.cast(ph, tf.complex128)
    mask = tf.cast(mask, tf.complex128)

    ph_masked = ph * mask  # [batch, N, N]

    P = ft2(ph_masked, delta)  # FT{φ·mask}
    S = ft2(tf.math.real(ph_masked) ** 2, delta)  # FT{(φ·mask)^2}，注意取实部平方
    W = ft2(mask, delta)  # FT{mask}

    N = tf.shape(ph)[-1]
    delta_f = 1 / (tf.cast(N, tf.float64) * delta)

    w2 = ift2(W * tf.math.conj(W), delta_f)  # IFT{W·W*}
    sw = tf.math.real(S * tf.math.conj(W)) - tf.abs(P) ** 2  # Re{S·W*} - |P|²
    D = (2 * ift2(tf.cast(sw, tf.complex128), delta_f)) / (w2 + 1e-12) * mask  # 避免除0，加epsilon

    return tf.math.real(D)  # 输出实部 [batch, N, N]


def str_cont(w, ph):
    return 2 * w ** 2 * (1 - ph)


def plot_struct(z_coes_array):
    SIZE = 256
    phase_screen = PhaseScreen(N=SIZE)
    r = phase_screen.r
    for i, z_coes in tqdm(enumerate(z_coes_array), total=z_coes_array.shape[0],
                          desc="computing struct..."):
        phase_screen.set_zernike_coeffients(list(z_coes))
        # phase_screen.add_pupil()
        ph = phase_screen.get_screen()
        ph = tf.cast(tf.convert_to_tensor(ph), tf.complex128)

        mask = tf.cast(tf.convert_to_tensor(r < 1), tf.complex128)
        strn = abs(str_fcn2_ft(ph, mask, 1)).numpy()
        # plt.imshow(strn)
        # ================================
        # 1) 假设已有 2D 相位结构函数 Dphi_2d
        #    这里随机生成一个示例 (只作演示)
        Dphi_2d = strn  # 你实际应使用真正的结构函数数据

        # 2) 对 2D 数组做环平均
        radii_pix, Dphi_1d = radial_profile_2d_2(Dphi_2d, r)
        if i == 0:
            Dphi_1ds = np.zeros([z_coes_array.shape[0], Dphi_1d.shape[0]])
        Dphi_1ds[i] = Dphi_1d

    Dphi_1d_mean = np.mean(Dphi_1ds, axis=0)
    Dphi_1d_std = np.std(Dphi_1ds, axis=0)
    assert Dphi_1d_mean.shape == Dphi_1d.shape, "形状有误"
    # 3) 转换为 r/r0
    #    假设 1个像素 = 1个长度单位(可根据实际需求改成: r_phys = radii_pix * pixel_scale)
    r0 = 1  # 示例: 设定 Fried 参数 r0 = 10 (与像素同单位)
    r_over_r0 = radii_pix / 1
    return Dphi_1d_mean, Dphi_1d_std, r_over_r0

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


def plot_struct_batch(z_coes_array, batch_size=64, SIZE=SCREEN_SIZE, delta=1.0, r0=1.0):
    """
    批量计算相位结构函数的环平均，并求平均。

    参数:
        z_coes_array: [num_samples, znum] ndarray，Zernike系数数组
        batch_size: 每次处理多少张
        SIZE: 相位屏尺寸 (默认256)
        delta: 像素间距 (默认1)
        r0: Fried参数，用于归一化r/r0 (默认1)

    返回:
        Dphi_1d_mean: 平均后的1D相位结构函数
        r_over_r0: 归一化后的半径数组
    """
    # 初始化 BatchPhaseScreen
    phase_screen = BatchPhaseScreen(N=SIZE,batch=batch_size)
    r = phase_screen.r  # [N, N]，半径矩阵

    # 创建掩膜
    mask = (r < 1).astype(np.float64)
    mask_tf = tf.cast(tf.convert_to_tensor(mask), tf.complex128)  # TensorFlow版

    num_samples = z_coes_array.shape[0]

    Dphi_1ds = []  # 用于保存每一张的1D结构函数

    for start_idx in tqdm(range(0, num_samples, batch_size),
                          desc="Computing structure functions batch"):
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch = z_coes_array[start_idx:end_idx]  # [batch_now, znum]
        batch_now = current_batch.shape[0]

        # 设置当前batch的zernike系数并更新相位屏
        phase_screen.set_zernike_coeffients(current_batch,update_scr=True)

        # 获取当前batch相位屏
        ph_stack = tf.convert_to_tensor(phase_screen.scr_stack, dtype=tf.complex128)

        # 计算当前batch的结构函数 [batch_now, N, N]
        if mask_tf.shape.rank == 2:
            mask_batch = tf.broadcast_to(mask_tf, (batch_now, SIZE, SIZE))
        else:
            mask_batch = mask_tf  # 已经是 batch版

        D_batch = str_fcn2_ft_batch(ph_stack, mask_batch, delta)  # [batch_now, N, N]

        # 转成numpy
        D_batch_np = tf.math.abs(D_batch).numpy()

        # 对每张2D结构函数做环平均
        radii_pix, radial_means = radial_profile_2d_batch(D_batch_np, r)

        # 保存
        Dphi_1ds.append(radial_means)

    # 合并所有样本
    Dphi_1ds = np.concatenate(Dphi_1ds, axis=0)  # [num_samples, nbins]

    # 求平均
    Dphi_1d_mean = np.mean(Dphi_1ds, axis=0)
    Dphi_1d_std = np.std(Dphi_1ds, axis=0)

    # r/r0
    r_over_r0 = radii_pix / r0

    return Dphi_1d_mean, Dphi_1d_std, r_over_r0


def plot_struct_curve(
        r_over_r0,
        Dphi_1d_mean,
        Dphi_1d_std=None,
        loglog=True,
        figsize=(6, 4),
        title=None,
        xlabel=r'$r/r_0$',
        ylabel=r'$D_\phi(r)$',
        color_mean='blue',
        color_std='lightblue',
        alpha_std=0.3,
        grid=True,
        legend=True
):
    """
    绘制结构函数曲线
    参数：
        r_over_r0: [nbins] 半径/r0数组
        Dphi_1d_mean: [nbins] 平均结构函数
        Dphi_1d_std: [nbins] (可选)标准差
        loglog: 是否使用log-log坐标
        figsize: 图大小
        title: 图标题
        xlabel, ylabel: 轴标签
        color_mean: 曲线颜色
        color_std: 阴影颜色
        alpha_std: 阴影透明度
        grid: 是否加网格
        legend: 是否加图例
    """
    plt.figure(figsize=figsize)

    # 阴影区域
    if Dphi_1d_std is not None:
        plt.fill_between(r_over_r0,
                         Dphi_1d_mean - Dphi_1d_std,
                         Dphi_1d_mean + Dphi_1d_std,
                         color=color_std,
                         alpha=alpha_std,
                         label='Std Dev' if legend else None)

    # 平均曲线
    plt.plot(r_over_r0, Dphi_1d_mean, color=color_mean, label='Mean' if legend else None)

    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    if grid:
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if legend:
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ps = PhaseScreen(N=256)
    ps.simulate_turbulence(15)
    ph = ps.get_screen()
    l = 16
    N = 256
    delta = l / N
    x = np.linspace(-N / 2, N / 2) * delta
    x, y = np.meshgrid(x, x)
    w = 2
    F = 1 / l
    mask = np.ones_like(ph)
    D = str_fcn2_ft(ph, mask, 1)
    D_cont = str_cont(w, ph)
    plt.imshow(np.real(D))
    plt.colorbar()
    plt.show()
    plt.imshow(np.real(D_cont))
    plt.colorbar()
    plt.show()

    # plt.plot(r_over_r0, Dphi_1d_mean, 'bo-', label='Radial Dphi(r)')
    # plt.xlabel(r'$r / r_0$')
    # plt.ylabel(r'$D_\phi(r)$')
    # plt.title('1D radial structure function from 2D batch_data')
    # plt.grid(True)
    # plt.legend()
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
    plt.savefig("./SampledImgs/Kolmogorov.png")
