import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from scipy.stats import chi2
from sklearn.decomposition import PCA
from win32file import FD_OOB

from plot_struct import plot_struct

def compare_pca_spectrum(x, y, k=10):
    pca_x = PCA(n_components=k).fit(x)
    pca_y = PCA(n_components=k).fit(y)
    spectrum_x = pca_x.explained_variance_ratio_
    spectrum_y = pca_y.explained_variance_ratio_
    return np.linalg.norm(spectrum_x - spectrum_y)

def calculate_mean_cov(z):
    mean = np.mean(z, axis=0)
    cov = np.cov(z, rowvar=False)
    return mean, cov
class FD_calculator(object):
    def __init__(self,real_data,epsilon=1e-6):
        self.epsilon = epsilon
        self.real_cov = np.cov(real_data, rowvar=False)
        self.real_cov = np.eye(self.real_cov.shape[0]) * self.epsilon
        self.real_mean = np.mean(real_data)
    def frechet_distance(self,y,verbose=True):
        # assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        mu_y, cov_y = calculate_mean_cov(y)

        # Step 2: 添加正则项以增强协方差稳定性
        cov_y += np.eye(cov_y.shape[0]) * self.epsilon

        # Step 3: 矩阵乘积开方计算
        cov_prod = self.real_cov @ cov_y
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
        diff = self.real_mean - mu_y
        trace_term = np.trace(self.real_cov + cov_y - 2 * sqrt_prod)
        fid = diff.dot(diff) + trace_term

        return max(fid, 0.0)


def frechet_distance(x, y, epsilon=1e-6, verbose=True):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
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
        # print("Mean squared difference:", diff.dot(diff))
        # print("Trace term:", trace_term)
        # print("FID (before clip):", fid)
        ...

    return max(fid, 0.0)
def get_common_xy_limits(*datasets_2d, margin=0.1):
    all_data = np.vstack(datasets_2d)
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


def add_confidence_ellipse(ax, data, confidence=0.95, color='black', label=None, linestyle='--'):
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(vals * chi2_val)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=2,
                      label=label)
    ax.add_patch(ellipse)


def compare_FD(real_data, gan_data, ddpm_data, csv_name, confidence=0.95, name=''):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    # 将所有数据合并用于统一 PCA 降维
    # all_data = np.vstack([real_data, gan_data, ddpm_data, ddim_data])
    # pca = PCA(n_components=2)
    # all_2d = pca.fit_transform(all_data)
    #
    # # 拆分降维后的数据
    # n_real = len(real_data)
    # n_gan = len(gan_data)
    # n_ddpm = len(ddpm_data)
    # real_2d = all_2d[:n_real]
    # gan_2d = all_2d[n_real:n_real + n_gan]
    # ddpm_2d = all_2d[n_real + n_gan:n_real + n_gan + n_ddpm]
    # ddim_2d = all_2d[n_real + n_gan + n_ddpm:]
    #
    # xlim, ylim = get_common_xy_limits(real_2d, gan_2d, ddpm_2d, ddim_2d)
    #
    # # 画图：三组比较
    # fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
    # for ax, gen_2d, title, color in zip(
    #     axs,
    #     [gan_2d, ddpm_2d, ddim_2d],
    #     ['GAN', 'DDPM', 'DDIM'],
    #     ['darkorange', 'steelblue', 'mediumseagreen']
    # ):
    #     ax.scatter(real_2d[:, 0], real_2d[:, 1], s=10, c='gray', alpha=0.4, label='Real')
    #     ax.scatter(gen_2d[:, 0], gen_2d[:, 1], s=15, c=color, alpha=0.6, label=title)
    #     add_confidence_ellipse(ax, real_2d, confidence, color='gray', label='Real 95% CI')
    #     add_confidence_ellipse(ax, gen_2d, confidence, color=color, label=f'{title} 95% CI')
    #     ax.set_title(f'{title} vs Real')
    #     ax.set_xlim(*xlim)
    #     ax.set_ylim(*ylim)
    #     ax.set_xlabel('PCA 1')
    #     ax.set_ylabel('PCA 2')
    #     ax.legend()
    #
    # plt.suptitle('PCA Visualization with 95% Confidence Ellipses: GAN vs DDPM vs DDIM', fontsize=16)
    # plt.tight_layout(rect=[0., 0.03, 1., 0.95])
    # plt.savefig(f'./Record/PCA{name}.png', dpi=300)

    # 计算 Fréchet Distance
    GAN_FD = frechet_distance(real_data, gan_data)
    DDPM_FD = frechet_distance(real_data, ddpm_data)
    # DDIM_FD = frechet_distance(real_data, ddim_data)
    result=pd.DataFrame([{"GAN_FD":GAN_FD,"ddpm_FD":DDPM_FD}])
    if not os.path.exists(csv_name):
        result.to_csv(csv_name, index=False)
    else:
        result.to_csv(csv_name, mode='a', index=False,header=False)
    print(f'GAN_FD: {GAN_FD:.4f},  ddpm_FD: {DDPM_FD:.4f}')

def compare_structure(real_data, ddpm_data, gan_data,maxlen,csv_name,params_dict,name=''):
    def nrmse(real_struct, fake_struct):
        if real_struct.shape[0] != fake_struct.shape[0]:

            maxlen = min(real_struct.shape[0], fake_struct.shape[0])
        else:
            maxlen = real_struct.shape[0]
        A=np.square(real_struct[:maxlen]-fake_struct[:maxlen])
        B=np.square(real_struct[:maxlen])
        # print(model_struct'A{A.shape,A},B{B.shape,B}')
        result=np.sqrt(np.sum(A)/np.sum(B))
        return result

    D_mean_ddpm,D_std_ddpm, r_over_r0_ddpm = plot_struct(ddpm_data[:maxlen],R=params_dict['R'],r0=params_dict['r0'])
    D_mean_gan, D_std_gan,r_over_r0_gan = plot_struct(gan_data[:maxlen],R=params_dict['R'],r0=params_dict['r0'])
    D_mean,D_std, r_over_r0 = plot_struct(real_data[:maxlen],R=params_dict['R'],r0=params_dict['r0'])
    fig = plt.figure(figsize=(6, 4))
    plt.plot(r_over_r0, D_mean, 'go-', label='ground truth $D_\phi(r)$')
    # plt.fill_between(r_over_r0, D_mean-D_std,D_mean+D_std,color='green',alpha=0.3)
    plt.plot(r_over_r0_ddpm, D_mean_ddpm, 'ro-', label='ddpm sampled $D_\phi(r)$')
    # plt.fill_between(r_over_r0, D_mean_ddpm-D_std_ddpm,D_mean_ddpm+D_std_ddpm,color='red',alpha=0.3)

    plt.plot(r_over_r0_gan, D_mean_gan, 'bo-', label='gan sampled $D_\phi(r)$')
    # plt.fill_between(r_over_r0, D_mean_gan-D_std_gan,D_mean_gan+D_std_gan,color='blue',alpha=0.3)
    plt.xlabel(r'$r / r_0$')
    plt.ylabel(r'$D_\phi(r)$')
    plt.title('1D radial structure function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Record/compare_struct{name}.png", dpi=300)
    plt.close(fig)
    # 计算NRMSE

    nrmse_ddpm = nrmse(D_mean, D_mean_ddpm)
    nrmse_gan = nrmse(D_mean, D_mean_gan)
    # nrmse_ddim = nrmse(D_mean, D_mean_ddim)
    print(f'GAN_NRMSE: {nrmse_gan:.4f}, ddpm_nrmse: {nrmse_ddpm:.4f}')
    result=pd.DataFrame([{"GAN_NRMSE": nrmse_gan,"ddpm_nrmse": nrmse_ddpm}])
    if not os.path.exists(csv_name):
        result.to_csv(csv_name,index=False)
    else:
        result.to_csv(csv_name,header=False,index=False,mode='a')
    np.savez(f'./Record/struct{name}',D_mean=D_mean,D_std=D_std,
                                         D_mean_gan=D_mean_gan,D_std_gan=D_std_gan,
                                         D_mean_ddpm=D_mean_ddpm,D_std_ddpm=D_std_ddpm,)

    # todo: 样本数-NRMSE图
    # fig1 = plt.figure(figsize=(6, 4))
    # plt.plot(r_over_r0, nrmse_ddpm, 'r--', label='DDPM $NRMSE$')
    # plt.plot(r_over_r0, nrmse_gan, 'b-', label='GAN $NRMSE$')
    # plt.xlabel(r'$r / r_0$')
    # plt.ylabel(r'$NRMSE (%)$')
    # plt.title('NRMSE between sampled batch_data and ground truth')
    # plt.tight_layout()
    # plt.savefig("./SampledImgs/NRMSE.png", dpi=300)
    # plt.close(fig1)

def compare_pca_norm(real_data,gan_data,ddpm_data,csv_name):
    ddpm_norm=compare_pca_spectrum(real_data,ddpm_data)
    # ddim_norm=compare_pca_spectrum(real_data,ddim_data)
    gan_norm=compare_pca_spectrum(real_data,gan_data)
    result=pd.DataFrame([{"GAN_norm":gan_norm,"DDPM_norm":ddpm_norm}])
    if not os.path.exists(csv_name):
        result.to_csv(csv_name,index=False)
    else:
        result.to_csv(csv_name,header=False,index=False,mode='a')
    print(f'GAN_norm:{gan_norm:.4f}, DDPM_norm:{ddpm_norm:.4f}')

if __name__ == '__main__':
    real_data=pandas.read_pickle('./Datasets/merged_data-10~15.pkl')
    real_data = np.stack(real_data, axis=0).astype(np.float32)
    ddpm_data=np.load('./Datasets/samples_ddpm-10~15.npy')
    ddim_data=np.load('./Datasets/samples_ddim-10~15.npy')
    gan_data=np.load('./Datasets/samples_gan-10~15.npy')
    compare_structure(real_data,ddpm_data=ddpm_data,gan_data=gan_data,maxlen=5000)
    compare_FD(real_data, gan_data=gan_data, ddpm_data=ddpm_data, csv_name='')
    compare_pca_norm(real_data,gan_data=gan_data,ddpm_data=ddpm_data,csv_name='')
    # compare_speed(sample_num=200, N=1024, batch_size=1)

