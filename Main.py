from Train import DDPMTrainer
from config import MODEL_CONFIG
from gan import GANHelper
from VGG import VGG16Trainer
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
    DDPM_trainer.train()
    DDPM_trainer.summary()

    gan_helper=GANHelper()
    gan_helper.train()

    # VGG_trainer = VGG16Trainer(model_config=MODEL_CONFIG)
    # VGG_trainer.train_model(epochs=100)

