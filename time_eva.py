import os
import time
import pandas as pd
from tqdm import tqdm
from PHASE_SCREEN import timing_per_screen
from PSD import Kolmogorov
from Train import DDPMTrainer
from aberration import PhaseScreen
from config import MODEL_CONFIG
from gan import GANHelper
import tensorflow as tf

def compare_speed(sample_num,N,batch_size):
    DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG,data_path=f'./Datasets/train-param1-500.npy')
    gan_helper=GANHelper(data_path=f'./Datasets/train-param1-500.npy')
    init_start=time.time()
    samples_ddpm = DDPM_trainer.ddpm.generate_samples(sample_num).numpy()
    phase_screen = PhaseScreen(N=N,)
    init_end=time.time()
    init_time = init_end - init_start

    print(f'Zernike polynomial time:{init_time}')
    # timing-ddpm
    print("start")
    start_time = time.time()
    samples_ddpm = DDPM_trainer.ddpm.generate_samples(sample_num).numpy()
    end_time_generated_coeff = time.time()
    sample_num = samples_ddpm.shape[0]
    for start_idx in tqdm(range(0, sample_num, batch_size),
                          desc="convert phase screen batch"):
        end_idx = min(start_idx + batch_size, sample_num)
        current_batch = samples_ddpm  # [batch_now, znum]
        batch_now = current_batch.shape[0]
        # 设置当前batch的zernike系数并更新相位屏
        for coeff in current_batch:
            phase_screen.set_zernike_coeffients(coeff,update_scr=True)
    end_time=time.time()
    time_per_screen_DDPM = (end_time-start_time)/sample_num
    time_per_coeff_DDPM = (end_time_generated_coeff-start_time)/sample_num
    print(f'Time per screen-DDPM: {time_per_screen_DDPM:.5f}')
    print(f'Time per coeff-DDPM:{time_per_coeff_DDPM:.5f}')
    # timing-gan
    start_time = time.time()
    samples_gan = gan_helper.gan.generate_samples(sample_num).numpy()
    end_time_generated_coeff = time.time()
    sample_num = samples_gan.shape[0]
    for start_idx in tqdm(range(0, sample_num, batch_size),
                          desc="convert phase screen batch"):
        end_idx = min(start_idx + batch_size, sample_num)
        current_batch = samples_ddpm  # [batch_now, znum]
        batch_now = current_batch.shape[0]
        # 设置当前batch的zernike系数并更新相位屏
        for coeff in current_batch:
            phase_screen.set_zernike_coeffients(coeff,update_scr=True)
    end_time = time.time()
    time_per_screen_GAN = (end_time - start_time) / sample_num
    time_per_coeff_GAN = (end_time_generated_coeff-start_time)/sample_num
    print(f'Time per screen-gan: {time_per_screen_GAN:.5f}')
    print(f'Time per coeff-gan:{time_per_coeff_GAN:.5f}')
    # timing-fft-sh
    D=2
    dx=D/N
    sub_harm=4
    psd=Kolmogorov
    time_FFT=timing_per_screen(repeat_num=sample_num,N=N,dx=dx,psd=psd,sub_harm=sub_harm)
    csv_name='./Record/speed.csv'
    result=pd.DataFrame([{"GAN_ps_time":time_per_screen_GAN,"GAN_coeff_time":time_per_coeff_GAN,"DDPM_ps_time":time_per_screen_DDPM,"DDPM_coeff_time":time_per_coeff_DDPM,"FFT_time":time_FFT}])
    if not os.path.exists(csv_name):
        result.to_csv(csv_name, index=False)
    else:
        result.to_csv(csv_name, header=False, index=False, mode='a')
if __name__=='__main__':
    A=tf.constant([1.,2.])
    B=tf.constant([3.,4.])
    C=tf.reduce_sum(A)
    print(C)
    compare_speed(sample_num=250 ,N=4096,batch_size=128)
