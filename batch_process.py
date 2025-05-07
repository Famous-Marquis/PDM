import numpy as np
import pandas

from VGG import train_VGG
from metrics import compare_structure, compare_pca_norm, compare_models_with_pca

if __name__ == '__main__':
    # generate_data(length_per_Dr0=10, nums_Dr0=10,
    #               Dr0_range=list(np.linspace(10., 15., num=10, endpoint=True, dtype=float)))
    # pkl = pd.read_pickle("./Datasets/merged_data.pkl")
    # gen_img_label('./Datasets/test_data-10~15.pkl', 'test')
    # gen_img_label('./Datasets/samples_ddpm1-10~15.npy','ddpm1-10~15')
    # gen_img_label('./Datasets/samples_ddpm2-10~15.npy','ddpm2-10~15')
    # gen_img_label('./Datasets/samples_ddpm3-10~15.npy', 'ddpm3-10~15')
    #
    # gen_img_label('./Datasets/samples_gan1-10~15.npy','gan1-10~15')
    # gen_img_label('./Datasets/samples_gan2-10~15.npy','gan2-10~15')
    # gen_img_label('./Datasets/samples_gan3-10~15.npy','gan3-10~15')
    # gen_img_label('./Datasets/merged_data-10~15.pkl','real')
    # gen_data((25, 1, [5, 6], 1, 1))
    ################################################################
    real_data = pandas.read_pickle('./Datasets/merged_data-10~15.pkl')
    real_data = np.stack(real_data, axis=0).astype(np.float32)
    for i in ['1', '2', '3']:
        ddpm_data = np.load(f'./Datasets/samples_ddpm{i}-10~15.npy')
        ddim_data = np.load(f'./Datasets/samples_ddim{i}-10~15.npy')
        gan_data = np.load(f'./Datasets/samples_gan{i}-10~15.npy')
        compare_structure(real_data, ddpm_data=ddpm_data, gan_data=gan_data, ddim_data=ddim_data,
                          maxlen=5000, csv_name='./Record/nrmse-10~15.csv')
        compare_models_with_pca(real_data, gan_data=gan_data, ddpm_data=ddpm_data,
                                ddim_data=ddim_data, csv_name='./Record/FD-10~15.csv')
        compare_pca_norm(real_data, gan_data=gan_data, ddpm_data=ddpm_data, ddim_data=ddim_data,
                         csv_name='./Record/norm-10~15.csv')

    train_VGG(data_name='ddpm1-10~15', test_data_name='test-10~15', name='ddpm1-10~15')
    train_VGG(data_name='ddpm2-10~15', test_data_name='test-10~15', name='ddpm2-10~15')
    train_VGG(data_name='ddpm3-10~15', test_data_name='test-10~15', name='ddpm3-10~15')
    train_VGG(data_name='gan1-10~15', test_data_name='test-10~15', name='gan1-10~15')
    train_VGG(data_name='gan2-10~15', test_data_name='test-10~15', name='gan2-10~15')
    train_VGG(data_name='gan3-10~15', test_data_name='test-10~15', name='gan3-10~15')
    train_VGG(data_name='real-10~15', test_data_name='test-10~15', name='1real-10~15')
    train_VGG(data_name='real-10~15', test_data_name='test-10~15', name='2real-10~15')
    train_VGG(data_name='real-10~15', test_data_name='test-10~15', name='3real-10~15')
