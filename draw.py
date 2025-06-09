import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd

from aberration import PhaseScreen, BatchPhaseScreen
from generate_data import process_coeff_beam, beam_list
from plot_struct import plot_struct

if __name__ == '__main__':
    ddpm_data = np.load(f'./Datasets/samples_ddpm1-10~15-s.npy')
    x,y, r_over_r0=plot_struct(ddpm_data[:10])
    with np.load('./Record/struct2-10~15-ss.npz') as data:
        D_mean_gan=data['D_mean_gan']
        D_mean_ddpm=data['D_mean_ddpm']
        D_mean=data['D_mean']
    plt.plot(r_over_r0,D_mean_ddpm,'ro-',label='PS-DDPM')
    plt.plot(r_over_r0,D_mean_gan,'co-',label='GAN')
    plt.plot(r_over_r0,D_mean,'bo-',label='ground truth')
    plt.legend()
    plt.xlabel(r'$r / r_0$')
    plt.ylabel(r'$D_\phi(r)$')
    plt.title('structure function')
    plt.grid(True)
    plt.show()

    gan_FD=[]
    ddpm_FD=[]

    gan_nrmse=[]
    ddpm_nrmse=[]
    sample_num=[250,500,1500,5000]
    for size in ['ss','s','m','l']:
        df=pandas.read_csv(f'./Record/FD-10~15-{size}.csv')
        gan_data=df['GAN_FD']
        DDPM_data=df['DDPM_FD']
        gan_FD.append(gan_data.mean())
        ddpm_FD.append(DDPM_data.mean())
    for size in ['ss','s','m','l']:
        df=pandas.read_csv(f'./Record/nrmse-10~15-{size}.csv')
        gan_data=df['GAN_NRMSE']
        DDPM_data=df['DDPM_NRMSE']
        gan_nrmse.append(gan_data.mean())
        ddpm_nrmse.append(DDPM_data.mean())
    # fig,ax=plt.subplots(1, 2, figsize=(10, 5))
    #
    # ax[0].grid()
    # ax[0].plot(sample_num,gan_FD,'r-',label='GAN')
    # ax[0].plot(sample_num, ddpm_FD,'b-',label='DDPM')
    # ax[0].legend()
    # # ax[0].set_xscale('log')
    # ax[0].set_xlabel('sample number')
    # ax[0].set_ylabel('FD')
    #
    # ax[1].grid()
    # ax[1].plot(sample_num,np.array(gan_nrmse)*100,'r-',label='GAN')
    # ax[1].plot(sample_num, np.array(ddpm_nrmse)*100,'b-',label='DDPM')
    # ax[1].legend()
    # # ax[1].set_xscale('log')
    # ax[1].set_xlabel('sample number')
    # ax[1].set_ylabel('NRMSE(%)')
    #
    # plt.suptitle('Comparison between GAN and DDPM')
    # plt.tight_layout()
    # plt.show()

    print(f'GAN_FD: {gan_FD}')
    print(f'DDPM_FD: {ddpm_FD}')
    print(f'GAN_NRMSE: {gan_nrmse}')
    print(f'DDPM_NRMSE: {ddpm_nrmse}')

    df=pd.read_csv('./Record/speed.csv')
    gan_time=df['GAN_time']
    ddpm_time=df['DDPM_time']
    FFT_time=df['FFT_time']
    size=[128,256,512,1024,2048]
    plt.plot(size,np.array(ddpm_time)*1000,'ro-',label='PS-DDPM')
    plt.plot(size,np.array(gan_time)*1000,'co-',label='GAN')
    plt.plot(size,np.array(FFT_time)*1000,'bo-',label='FFT-SH')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pixel Size')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.title('Computational Efficiency')
    plt.show()
    ps=BatchPhaseScreen(1)
    beam=beam_list[7]
    plt.imshow(beam)
    plt.show()
    for i in range(10):
        ps.simulate_turbulence(1)
        coeff=ps.get_coeffients()
        img=process_coeff_beam(coeff,beam,ps)
        plt.imshow(img)
        plt.show()
