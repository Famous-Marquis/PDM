import gc
import os

import numpy as np
from keras.backend import clear_session
import tensorflow as tf
from Train import DDPMTrainer
from VGG import train_VGG
from aberration import PhaseScreen
from config import MODEL_CONFIG
from gan import GANHelper
from generate_data import gen_img_label, gen_data
from metrics import compare_structure, compare_FD

# params = {
#     'r0': 0.05, 'l0': 5e-3, 'L0': 10, "R": 0.5, "alpha": 11 / 3
# }
params = {
    'r0': 0.05, 'l0': 5e-3, 'L0': 10, "R": 0.5, "alpha": 14 / 3
}
# params = {
#     'r0': 0.05, 'l0': 5e-3, 'L0': 10, "R": 0.5, "alpha": 10 / 3
# }

if __name__ == '__main__':
    size = "2000" #记得该名称
    length = 2000 #需要训练5000，1000，500的样本大小
    # s = pandas.read_pickle('./Datasets/merged_data-param3-s.pkl')
    # s = np.stack(s).shape
    # print(s)
    ps = PhaseScreen()
    ps.simulate_turbulence(r0=params['r0'], l0=params['l0'], L0=params['L0'], R=params['R'],
             alpha=params['alpha'])
    # 更改了数据集之后，需要重新生成数据集
    gen_data(r0=params['r0'], l0=params['l0'], L0=params['L0'], R=params['R'],
             alpha=params['alpha'], ps=ps, nums=100, name=f'./Datasets/test-param3-{size}.npy')
    gen_data(r0=params['r0'], l0=params['l0'], L0=params['L0'], R=params['R'],
             alpha=params['alpha'], ps=ps, nums=length, name=f'./Datasets/train-param3-{size}.npy')
    # 训练模型
    for i in ['4', '5', '6']:
        DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG,
                                   data_path=f'./Datasets/train-param3-{size}.npy')
        DDPM_trainer.ddpm.build(input_shape=(MODEL_CONFIG["d_model"],))
        DDPM_trainer.train()
        DDPM_trainer.summary()

        gan_helper = GANHelper(data_path=f'./Datasets/train-param3-{size}.npy')
        gan_helper.train()
        try:
            raise FileNotFoundError
        except FileNotFoundError:
            samples_ddpm = DDPM_trainer.ddpm.generate_samples(2000).numpy()
            np.save(f"Datasets/samples_ddpm{i}-param3-{size}.npy", samples_ddpm)

            samples_gan = gan_helper.gan.generate_samples(2000).numpy()
            np.save(f"Datasets/samples_gan{i}-param3-{size}.npy", samples_gan)
            clear_session()
            gc.collect()
    #
    #
    # # 比较样本
    # ##############################################################
    real_data = np.load(f'./Datasets/train-param3-{size}.npy')
    #
    for i in ['4', '5', '6']:
        ddpm_data = np.load(f'./Datasets/samples_ddpm{i}-param3-{size}.npy')
        gan_data = np.load(f'./Datasets/samples_gan{i}-param3-{size}.npy')
        compare_structure(real_data, ddpm_data=ddpm_data, gan_data=gan_data,
                          maxlen=5000, csv_name=f'./Record/nrmse-param3-{size}.csv',
                          name=f'{i}-param3-{size}',params_dict=params)

        compare_FD(real_data, gan_data=gan_data, ddpm_data=ddpm_data,
                   csv_name=f'./Record/norm-param3-{size}.csv')
    # # 训练一下真实的样本

    gen_img_label(f'./Datasets/test-param3-{size}.npy', f'test-param3-{size}',ps=ps,num=100)
    gen_img_label(f'./Datasets/train-param3-{size}.npy', f'real-param3-{size}',ps=ps,num=500)
    for j in ['a','b','c']:
        for i in ['4','5','6',]:
            gen_img_label(f'./Datasets/samples_ddpm{i}-param3-{size}.npy', f'ddpm{i}-param3-{size}',ps=ps,num=1000)
            gen_img_label(f'./Datasets/samples_gan{i}-param3-{size}.npy', f'gan{i}-param3-{size}',ps=ps,num=1000)
            train_VGG(data_name=f'ddpm{i}-param3-{size}', test_data_name=f'test-param3-{size}',
                      name=f' {j} ddpm{i}-param3-{size}', kind='DDPM')
            train_VGG(data_name=f'gan{i}-param3-{size}', test_data_name=f'test-param3-{size}',
                      name=f' {j} gan{i}-param3-{size}', kind='GAN')
            train_VGG(data_name=f'real-param3-{size}', test_data_name=f'test-param3-{size}',
                      name=f' {j} {i}real-param3-{size}', kind='\"Real\"')

        # os.remove(f'./Datasets/VGG_Datasets_ddpm{i}-param3-{size}.npz')
        # os.remove(f'./Datasets/VGG_Datasets_gan{i}-param3-{size}.npz')
        # os.remove(f'./Datasets/VGG_Datasets_real{i}-param3-{size}.npz')
