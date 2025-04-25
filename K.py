import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2

from Train import frechet_distance


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
                      edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=2, label=label)
    ax.add_patch(ellipse)

def compare_models_with_pca(real_data, gan_data, ddpm_data, confidence=0.95):
    # 统一 PCA 降维
    all_data = np.vstack([real_data, gan_data, ddpm_data])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_data)

    n_real = len(real_data)
    n_gan = len(gan_data)
    real_2d = all_2d[:n_real]
    gan_2d = all_2d[n_real:n_real + n_gan]
    ddpm_2d = all_2d[n_real + n_gan:]

    xlim, ylim = get_common_xy_limits(real_2d, gan_2d, ddpm_2d)

    # 开始画图
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, gen_2d, title, color in zip(
        axs, [gan_2d, ddpm_2d], ['GAN', 'DDPM'], ['darkorange', 'steelblue']
    ):
        ax.scatter(real_2d[:, 0], real_2d[:, 1], s=10, c='gray', alpha=0.4, label='Real')
        ax.scatter(gen_2d[:, 0], gen_2d[:, 1], s=15, c=color, alpha=0.6, label=title)
        add_confidence_ellipse(ax, real_2d, confidence, color='gray', label='Real 95% CI')
        add_confidence_ellipse(ax, gen_2d, confidence, color=color, label=f'{title} 95% CI')
        ax.set_title(f'{title} vs Real')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.legend()

    plt.suptitle('PCA Visualization with 95% Confidence Ellipses: GAN vs DDPM', fontsize=14)
    plt.tight_layout(rect=[0., 0.03, 1., 0.95])
    plt.show()
    FD_gan=frechet_distance(real_data, gan_data)
    FD_ddpm=frechet_distance(real_data, ddpm_data)
    print(f'FD-GAN:{FD_gan},FD-DDPM:{FD_ddpm}')

def plot_coverage_pca_with_ellipses(
    real_data, fake_data, kmeans, covered_mask,
    confidence_level=0.95,  # 椭圆置信度（常用95%或99%）
    real_style={'color': 'gray', 'alpha': 0.4, 's': 15, 'marker': 'o', 'label': 'Real'},
    fake_style={'color': 'darkorange', 'alpha': 0.6, 's': 25, 'marker': 'o', 'label': 'Generated'},
    covered_color='green', uncovered_color='red'
):
    # PCA 降维
    all_data = np.vstack([real_data, fake_data, kmeans.cluster_centers_])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_data)

    n_real, n_fake, n_centers = len(real_data), len(fake_data), len(kmeans.cluster_centers_)
    real_2d = all_2d[:n_real]
    fake_2d = all_2d[n_real:n_real + n_fake]
    centers_2d = all_2d[-n_centers:]

    # 画图
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')

    # 画样本点
    plt.scatter(real_2d[:, 0], real_2d[:, 1], **real_style)
    plt.scatter(fake_2d[:, 0], fake_2d[:, 1], **fake_style)

    # 画聚类中心
    for i, (x, y) in enumerate(centers_2d):
        color = covered_color if covered_mask[i] else uncovered_color
        plt.scatter(x, y, c=color, s=120, edgecolors='black', marker='X',
                    label='Covered' if (color == covered_color and i == 0) else
                          'Uncovered' if (color == uncovered_color and i == 0) else None)

    # 添加置信椭圆
    def add_ellipse(points, color, label):
        cov = np.cov(points, rowvar=False)
        mean = np.mean(points, axis=0)

        # 特征值/向量 → 主轴
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # 椭圆轴长根据置信度的卡方值确定
        chi2_val = chi2.ppf(confidence_level, df=2)
        width, height = 2 * np.sqrt(vals * chi2_val)

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                        edgecolor=color, facecolor='none', linewidth=2, label=label, linestyle='--')
        plt.gca().add_patch(ellip)

    add_ellipse(real_2d, color=real_style['color'], label=f"{real_style['label']} 95% CI")
    add_ellipse(fake_2d, color=fake_style['color'], label=f"{fake_style['label']} 95% CI")

    plt.title(f'PCA Visualization with {int(confidence_level*100)}% Confidence Ellipses')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # 假设你有 TensorFlow 张量
    real_tensor = tf.random.normal(shape=(1000, 15))  # 1000 个真实样本，15维Zernike
    fake_tensor = tf.random.normal(shape=(500, 15))  # 500 个生成样本

    # 转成 numpy 用于 sklearn
    real_data = real_tensor.numpy()
    fake_data = fake_tensor.numpy()
    # 计算覆盖率



