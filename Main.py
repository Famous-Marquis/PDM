import pandas
import tensorflow as tf
from metrics import compare_models_with_pca
from Train import DDPMTrainer
from config import MODEL_CONFIG, MODEL_CONFIG2
from gan import GANHelper
from VGG import VGG16Trainer
import numpy as np
from metrics import compare_structure
from tensorflow.keras.backend import clear_session
import gc
if __name__ == '__main__':
    size='m'
    for i in ['1']:
        # tf.compat.v1.enable_eager_execution()
        DDPM_trainer = DDPMTrainer(model_config=MODEL_CONFIG)
        DDPM_trainer.ddpm.build(input_shape=(MODEL_CONFIG["d_model"],))
        DDPM_trainer.train()
        DDPM_trainer.summary()

        # DDIM_trainer = DDPMTrainer(model_config=MODEL_CONFIG2)
        # DDIM_trainer.ddpm.build(input_shape=(MODEL_CONFIG2["d_model"],))
        # DDIM_trainer.train()
        # DDIM_trainer.summary()
        #
        # gan_helper = GANHelper()
        # gan_helper.train()
        try:
            # samples_ddpm = np.load("Datasets/samples_ddpm.npy")
            # samples_gan=np.load("Datasets/samples_gan.npy")
            raise FileNotFoundError
        except FileNotFoundError:
            samples_ddpm = DDPM_trainer.ddpm.generate_samples(5000).numpy()
            np.save(f"Datasets/samples_ddpm{i}-10~15-{size}.npy", samples_ddpm)
            # samples_ddim = DDIM_trainer.ddpm.generate_samples(5000).numpy()
            # np.save(f"Datasets/samples_ddim{i}-10~15-{size}.npy", samples_ddim)
            # samples_gan = gan_helper.gan.generate_samples(5000).numpy()
            # np.save(f"Datasets/samples_gan{i}-10~15-{size}.npy", samples_gan)
            clear_session()
            gc.collect()
    # data_series = pandas.read_pickle(MODEL_CONFIG["data_path"])
    # real_samples = np.stack(data_series).astype(np.float32)
    # compare_models_with_pca(real_samples, samples_gan, samples_ddpm)
    # compare_structure(real_samples, samples_gan, samples_ddpm)
    # coverage, covered_k, used_delta = compute_coverage(real_samples, samples_ddpm, K=10)
    # coverage, covered_k, used_delta = compute_coverage(real_samples, samples_gan, K=10)
    # compute_coverage(real_samples, real_samples[:3000], K=10)
    # VGG_trainer = VGG16Trainer(model_config=MODEL_CONFIG)
    # VGG_trainer.train_model(epochs=100)

