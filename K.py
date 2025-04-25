import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_coverage_pca(real_data, fake_data, kmeans, covered_mask):
    """
    画出PCA投影下的覆盖热图。

    参数：
    - real_data: np.ndarray (N_real, D)
    - fake_data: np.ndarray (N_fake, D)
    - kmeans: 训练好的KMeans对象（用于获取聚类中心）
    - covered_mask: bool array, shape=(K,), 标记每个中心是否被覆盖
    """
    # 拼接所有数据用于PCA
    all_data = np.vstack([real_data, fake_data, kmeans.cluster_centers_])

    # PCA降维到2D
    pca = PCA(n_components=2)
    all_data_2d = pca.fit_transform(all_data)
    n_real, n_fake, n_centers = len(real_data), len(fake_data), len(kmeans.cluster_centers_)

    real_2d = all_data_2d[:n_real]
    fake_2d = all_data_2d[n_real:n_real + n_fake]
    centers_2d = all_data_2d[-n_centers:]

    # 设置画图风格
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')

    # 画真实样本
    plt.scatter(real_2d[:, 0], real_2d[:, 1], c='green', label='Real Samples', alpha=0.5, s=10)

    # 画生成样本
    plt.scatter(fake_2d[:, 0], fake_2d[:, 1], c='cornflowerblue', label='Generated Samples',
                alpha=0.6, s=15)

    # 画聚类中心
    for i, center in enumerate(centers_2d):
        color = 'green' if covered_mask[i] else 'red'
        plt.scatter(center[0], center[1], c=color, edgecolors='black', s=100, marker='X',
                    label='Covered' if (color == 'green' and i == 0) else 'Not Covered' if (
                                color == 'red' and i == 0) else None)
    # 图例
    plt.title('PCA Visualization with Cluster Center Coverage')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_coverage(real_data, fake_data, K=50, delta=None):
    """
    计算生成数据对真实数据聚类中心的覆盖率（mode coverage）

    参数:
    - real_data: np.ndarray, shape (N_real, D), 真实样本（如Zernike系数）
    - fake_data: np.ndarray, shape (N_fake, D), 生成样本
    - K: int, 聚类中心个数
    - delta: float or None, 距离阈值，若为None将自动设定

    返回:
    - coverage_rate: float, 覆盖率（0~1）
    - covered_clusters: int, 被覆盖的聚类中心个数
    - delta: 使用的距离阈值
    """
    assert real_data.shape[1] == fake_data.shape[1], "维度不一致"

    # Step 1: 对真实样本进行 KMeans 聚类
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(real_data)
    centers = kmeans.cluster_centers_  # shape: (K, D)

    # Step 2: 计算每个中心与所有生成样本之间的欧几里得距离
    dists = pairwise_distances(centers, fake_data, metric='euclidean')  # shape: (K, N_fake)

    # Step 3: 自动设定 delta（默认是真实数据的平均距离的 10%）
    if delta is None:
        real_pairwise_dists = pairwise_distances(real_data, metric='euclidean')
        avg_dist = np.mean(real_pairwise_dists)
        delta = 0.1 * avg_dist

    # Step 4: 判断是否被覆盖
    covered = (dists < delta).any(axis=1)  # 每行表示某中心是否被至少一个生成样本覆盖
    coverage_rate = np.sum(covered) / K
    print(f"覆盖率: {coverage_rate:.3f} ({np.sum(covered)}/{K} 个簇), 使用的阈值 delta = {delta:.4f}")
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(real_data)
    centers = kmeans.cluster_centers_

    # 2. 计算中心与 fake 样本的距离判断覆盖

    dists = pairwise_distances(centers, fake_data)
    delta = 0.1 * np.mean(pairwise_distances(real_data))
    covered_mask = (dists < delta).any(axis=1)

    # 3. 画图
    plot_coverage_pca(real_data, fake_data, kmeans, covered_mask)
    return coverage_rate, np.sum(covered), delta
if __name__ == '__main__':
    # 假设你有 TensorFlow 张量
    real_tensor = tf.random.normal(shape=(1000, 15))  # 1000 个真实样本，15维Zernike
    fake_tensor = tf.random.normal(shape=(500, 15))  # 500 个生成样本

    # 转成 numpy 用于 sklearn
    real_data = real_tensor.numpy()
    fake_data = fake_tensor.numpy()
    # 计算覆盖率
    coverage, covered_k, used_delta = compute_coverage(real_data, fake_data, K=50)



