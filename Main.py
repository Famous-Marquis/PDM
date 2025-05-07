import pandas
import tensorflow as tf
from metrics import compare_models_with_pca
from Train import DDPMTrainer
from config import MODEL_CONFIG, MODEL_CONFIG2
from gan import GANHelper
from VGG import VGG16Trainer
import numpy as np
from metrics import compare_structure

if __name__ == '__main__':
    # # 示例数据
    # x = np.random.rand(100, 2)  # 100 samples, 2 features
    # y = np.random.rand(100, 2)  # 100 samples, 2 features
    #
    # # 计算 FID
    # fd = Frechet_distance(x, y)
    # print(f"FID: {fd}")
    # DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG)
    # DDPM_trainer.ddpm.build(input_shape=(MODEL_CONFIG["d_model"],))
    # DDPM_trainer.train()
    # DDPM_trainer.summary()
    #
    # DDIM_trainer = DDPMTrainer(model_config=MODEL_CONFIG2)
    # DDIM_trainer.ddpm.build(input_shape=(MODEL_CONFIG2["d_model"],))
    # DDIM_trainer.train()
    # DDIM_trainer.summary()
    #
    # DDPM_trainer2 = DDPMTrainer(model_config=MODEL_CONFIG)
    # DDPM_trainer2.ddpm.build(input_shape=(MODEL_CONFIG["d_model"],))
    # DDPM_trainer2.train()
    # DDPM_trainer2.summary()
    #
    # DDIM_trainer2 = DDPMTrainer(model_config=MODEL_CONFIG2)
    # DDIM_trainer2.ddpm.build(input_shape=(MODEL_CONFIG2["d_model"],))
    # DDIM_trainer2.train()
    # DDIM_trainer2.summary()
    #
    # gan_helper=GANHelper()
    # gan_helper.train()

    gan_helper2=GANHelper()
    gan_helper2.train()
    try:
        # samples_ddpm = np.load("Datasets/samples_ddpm.npy")
        # samples_gan=np.load("Datasets/samples_gan.npy")
        raise FileNotFoundError
    except FileNotFoundError:
        # samples_ddpm = DDPM_trainer.ddpm.generate_samples(5000).numpy()
        # np.save("Datasets/samples_ddpm2.npy", samples_ddpm)
        # samples_ddim = DDIM_trainer.ddpm.generate_samples(5000).numpy()
        # np.save("Datasets/samples_ddim2.npy", samples_ddim)
        # samples_ddpm = DDPM_trainer2.ddpm.generate_samples(5000).numpy()
        # np.save("Datasets/samples_ddpm3.npy", samples_ddpm)
        # samples_ddim = DDIM_trainer2.ddpm.generate_samples(5000).numpy()
        # np.save("Datasets/samples_ddim3.npy", samples_ddim)
        # samples_gan=gan_helper2.gan.generate_samples(5000).numpy()
        # np.save("Datasets/samples_gan2.npy", samples_gan)
        samples_gan = gan_helper2.gan.generate_samples(5000).numpy()
        np.save("Datasets/samples_gan3.npy", samples_gan)
        ...


    # data_series = pandas.read_pickle(MODEL_CONFIG["data_path"])
    # real_samples = np.stack(data_series).astype(np.float32)
    # compare_models_with_pca(real_samples, samples_gan, samples_ddpm)
    # compare_structure(real_samples, samples_gan, samples_ddpm)
    # coverage, covered_k, used_delta = compute_coverage(real_samples, samples_ddpm, K=10)
    # coverage, covered_k, used_delta = compute_coverage(real_samples, samples_gan, K=10)
    # compute_coverage(real_samples, real_samples[:3000], K=10)
    # VGG_trainer = VGG16Trainer(model_config=MODEL_CONFIG)
    # VGG_trainer.train_model(epochs=100)

