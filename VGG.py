import os

import numpy as np
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, \
    Flatten, Dense

from generate_data import beam_list


class VGG16(keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(len(beam_list), activation='softmax')

    def call(self, x):
        assert len(x.shape) == 3, "x含有通道维度"
        x = x[:, None]
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


class VGG16Trainer:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = VGG16()
        self.data_path = self.model_config["VGG_datapath"]
        self.batch_size = model_config["batch_size"]
        self.ckpt_path = model_config["VGG_checkpoint_path"]
    def build_model(self):
        self.model.build(input_shape=(self.model_config['batch_size'], 224, 224))
        self.model.summary()

    def compile_model(self):
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.model_config['VGG_lr'],
            decay_rate=0.96, decay_steps=100)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    def plot_history(self, history_dict):
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(history_dict['loss'], label='train_loss')
        axes[0].plot(history_dict['val_loss'], label='test_loss')
        axes[0].legend()
        axes[0].title.set_text('loss')

        axes[1].plot(history_dict['sparse_categorical_accuracy'], label='train_acc')
        axes[1].plot(history_dict['val_sparse_categorical_accuracy'], label='test_acc')
        axes[1].legend()
        axes[1].title.set_text('acc')

        plt.savefig('./Record/VGG16_history.png')

    def train_model(self,epochs):
        self.build_model()
        self.compile_model()
        cp_callback = ModelCheckpoint(filepath=self.ckpt_path, save_best_only=True,
                                      save_weights_only=True, )
        if os.path.exists(self.ckpt_path + '.index'):
            print("model load weights")
            model.load_weights(self.ckpt_path)
        else:
            print("No model weights")
        data = np.load(self.model_config["VGG_datapath"])
        imgs_train, labels_train = data["imgs"], data["labels"]
        imgs_val, labels_val = ..., ...
        history=model.fit(imgs_train, labels_train, batch_size=self.batch_size, epochs=epochs,
                  validation_data=(imgs_val, labels_val),callbacks=[cp_callback])
        history_dict = history.history

    # VGG比较简单，可以直接用keras封装好的函数库


if __name__ == "__main__":
    model = VGG16()
    model.build((64, 224, 224))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    model.fit()
