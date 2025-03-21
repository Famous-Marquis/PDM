from aberration import ZERNIKE_NUMS

MODEL_CONFIG = {
    "predict_cov": False,  # True or False
    "cosine_schedule": True,
    "beta_1": 0.0001,
    "beta_T": 0.08, # 调节这两个参数，使得正向扩散的终点接近高斯噪声
    "T": 50,
    "epochs": 300,
    "GAN_epochs": 30,
    "batch_size": 64,
    "lr_min": 0.001,#0.01,
    "lr_max": 0.1,#0.3,
    "gan_learning_rate": 0.001,
    "model_checkpoint_path": "./Checkpoints/DDPM_model.ckpt",
    "gan_checkpoint_path": "./Checkpoints/gan.ckpt",
    "data_path": "./DDPM_data/merged_data.pkl",
    "VGG_data_path":None,# todo: VGG 数据库
    "d_model": ZERNIKE_NUMS,# 系数长度
    "load_weights": True,
    "GAN_load_weights": False,
    "model_mean_struct": [128,2048,4096],  # [List] 均值神经网络: 全连接层的神经元数
    "model_v_struct": [128, 256, 512, 512, 256],  # [List] 协方差神经网络: 全连接层的神经元数

    "generate_new_data": False,
    "length_per_Dr0": 2500,# 样本数 = length * nums_Dr0
    "nums_Dr0":2, # 生成数据的Dr0个数
    "Dr0_range":1,# List [min,max]  or CONST

    # 扩散过程
}