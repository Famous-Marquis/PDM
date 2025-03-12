from aberration import ZERNIKE_NUMS
from Train import DDPMTrainer
MODEL_CONFIG = {
    "predict_cov": False,  # True or False
    "cosine_schedule": True,
    "beta_1": 0.0001,
    "beta_T": 0.01,
    "T": 100,
    "epochs": 200,
    "batch_size": 64,
    "lr_min": 0.01,
    "lr_max": 0.3,
    "model_checkpoint_path": "./Checkpoints/DDPM_model.ckpt",
    "data_path": "./DDPM_data/merged_data.pkl",
    "d_model": ZERNIKE_NUMS,
    "load_weights": True,
    "model_mean_struct": [2048,4096],  # [List] 均值神经网络: 全连接层的神经元数
    "model_v_struct": [128, 256, 512, 512, 256],  # [List] 协方差神经网络: 全连接层的神经元数


    # 扩散过程
}
if __name__ == '__main__':
    # # 示例数据
    # x = np.random.rand(100, 2)  # 100 samples, 2 features
    # y = np.random.rand(100, 2)  # 100 samples, 2 features
    #
    # # 计算 FID
    # fd = Frechet_distance(x, y)
    # print(f"FID: {fd}")
    DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG)

    DDPM_trainer.ddpm.build(input_shape=(MODEL_CONFIG["d_model"],))
    DDPM_trainer.train(epochs=10)
    DDPM_trainer.summary()
