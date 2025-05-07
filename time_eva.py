import time

from tqdm import tqdm

from PHASE_SCREEN import timing_per_screen
from PSD import Kolmogorov
from Train import DDPMTrainer
from aberration import BatchPhaseScreen
from gan import GANHelper


def compare_speed(sample_num,N,batch_size):
    # todo: 一次性结束对不同N的比较
    DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG)
    gan_helper=GANHelper()
    init_start=time.time()
    phase_screen = BatchPhaseScreen(N=N,batch=batch_size)
    init_end=time.time()
    init_time = init_end - init_start
    print(f'Zernike polynomial time:{init_time}')
    samples_ddpm = DDPM_trainer.ddpm.generate_samples(sample_num).numpy()
    # timing-ddpm
    print("start")
    start_time = time.time()
    samples_ddpm = DDPM_trainer.ddpm.generate_samples(sample_num).numpy()
    end_time_generated_coeff = time.time()
    sample_num = samples_ddpm.shape[0]
    for start_idx in tqdm(range(0, sample_num, batch_size),
                          desc="Computing structure functions batch"):
        end_idx = min(start_idx + batch_size, sample_num)
        current_batch = samples_ddpm[start_idx:end_idx]  # [batch_now, znum]
        batch_now = current_batch.shape[0]
        # 设置当前batch的zernike系数并更新相位屏
        phase_screen.set_zernike_coeffients(current_batch,update_scr=True)
    end_time=time.time()
    time_per_screen = (end_time-start_time)/sample_num
    time_per_coeff = (end_time_generated_coeff-start_time)/sample_num
    print(f'Time per screen-DDPM: {time_per_screen:.3f}')
    print(f'Time per coeff-DDPM:{time_per_coeff:.3f}')
    # timing-gan
    start_time = time.time()
    samples_gan = gan_helper.gan.generate_samples(sample_num).numpy()
    end_time_generated_coeff = time.time()
    sample_num = samples_gan.shape[0]
    for start_idx in tqdm(range(0, sample_num, batch_size),
                          desc="Computing structure functions batch"):
        end_idx = min(start_idx + batch_size, sample_num)
        current_batch = samples_gan[start_idx:end_idx]  # [batch_now, znum]
        batch_now = current_batch.shape[0]
        # 设置当前batch的zernike系数并更新相位屏
        phase_screen.set_zernike_coeffients(current_batch, update_scr=True)
    end_time = time.time()
    time_per_screen = (end_time - start_time) / sample_num
    time_per_coeff = (end_time_generated_coeff-start_time)/sample_num
    print(f'Time per screen-gan: {time_per_screen:.3f}')
    print(f'Time per coeff-gan:{time_per_coeff:.3f}')
    # timing-fft-sh
    D=2
    dx=D/N
    sub_harm=4
    psd=Kolmogorov
    timing_per_screen(repeat_num=sample_num,N=N,dx=dx,psd=psd,sub_harm=sub_harm)