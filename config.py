import numpy as np

from aberration import ZERNIKE_NUMS

MODEL_CONFIG = {
    "model_name": 'DDPM',
    "predict_cov": False,  # True or False
    "cosine_schedule": False,
    "beta_1": 1e-5,#1e-5
    "beta_T": 0.035,  # 0.095调节这两个参数，使得正向扩散的终点接近高斯噪声
    "T": 50,
    "epochs": 200,
    "GAN_epochs": 200,
    "batch_size": 64,
    "lr_min": 0.01,  # 0.01,
    "lr_max": 0.3,  # 0.3,
    "gan_learning_rate": 0.0005,  # 0.0005
    "VGG_lr": 0.001,
    "model_checkpoint_path": "./Checkpoints/DDPM_model.ckpt",
    "gan_checkpoint_path": "./Checkpoints/gan.ckpt",
    "VGG_checkpoint_path": "./Checkpoints/VGG_model.ckpt",
    "VGG_data_path": None,
    "d_model": ZERNIKE_NUMS,  # 系数长度
    "load_weights": False,
    "GAN_load_weights": False,
    "model_mean_struct": [512,2048,4096],  # 256,1024,256/[List] 均值神经网络: 全连接层的神经元数
    "model_v_struct": [512, 2048],  # [List] 协方差神经网络: 全连接层的神经元数

    "generate_new_data": False,
    "length_per_Dr0": 10,  # 样本数 = length * nums_Dr0
    "nums_Dr0": 10,  # 生成数据的Dr0个数
    "Dr0_range": list(np.linspace(start=10., num=10, stop=15.0, endpoint=True)),
    # List [min,max]  or CONST
    # todo: 待DDPM与GAN训练好后，分别用DDPM与GAN生成对应数据库
    # 扩散过程
}
MODEL_CONFIG2 = MODEL_CONFIG.copy()
MODEL_CONFIG2['model_name'] = 'DDIM'
MODEL_CONFIG2['cosine_schedule'] = True
MODEL_CONFIG2['T'] = 20
MODEL_CONFIG2['model_checkpoint_path'] = "./Checkpoints/DDIM_model.ckpt"
MODEL_CONFIG2['load_weights'] = False
