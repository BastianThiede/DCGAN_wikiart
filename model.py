from keras.initializers import RandomNormal
from keras.layers import (Conv2DTranspose, Conv2D, BatchNormalization, Reshape,
                          Dense, LeakyReLU, ReLU, GaussianNoise, Dropout, Activation,
                          AveragePooling2D, Flatten, UpSampling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from utils import load_config


def generator():
    G = Sequential()

    G.add(Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

    # 1x1x4096
    G.add(Conv2DTranspose(filters=256, kernel_size=4))
    G.add(Activation('relu'))

    # 4x4x256 - kernel sized increased by 1
    G.add(Conv2D(filters=256, kernel_size=4, padding='same'))
    G.add(BatchNormalization(momentum=0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 8x8x256 - kernel sized increased by 1
    G.add(Conv2D(filters=128, kernel_size=4, padding='same'))
    G.add(BatchNormalization(momentum=0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 16x16x128
    G.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    G.add(BatchNormalization(momentum=0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 32x32x64
    G.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    G.add(BatchNormalization(momentum=0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 64x64x32
    G.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    G.add(BatchNormalization(momentum=0.7))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 128x128x16
    G.add(Conv2D(filters=8, kernel_size=3, padding='same'))
    G.add(Activation('relu'))
    G.add(UpSampling2D())

    # 256x256x8
    G.add(Conv2D(filters=3, kernel_size=3, padding='same'))
    G.add(Activation('sigmoid'))

    return G


def discriminator():
    D = Sequential()

    # add Gaussian noise to prevent Discriminator overfitting
    D.add(GaussianNoise(0.2, input_shape=[256, 256, 3]))

    # 256x256x3 Image
    D.add(Conv2D(filters=8, kernel_size=3, padding='same'))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 128x128x8
    D.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    D.add(BatchNormalization(momentum=0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 64x64x16
    D.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    D.add(BatchNormalization(momentum=0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 32x32x32
    D.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    D.add(BatchNormalization(momentum=0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 16x16x64
    D.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    D.add(BatchNormalization(momentum=0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 8x8x128
    D.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    D.add(BatchNormalization(momentum=0.7))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.25))
    D.add(AveragePooling2D())

    # 4x4x256
    D.add(Flatten())

    # 256
    D.add(Dense(128))
    D.add(LeakyReLU(0.2))

    D.add(Dense(1, activation='sigmoid'))
    print(D.summary())
    return D


def build_gan(config_path):
    config = load_config(config_path)
    g = generator()
    d = discriminator()
    opt = Adam(lr=config['learning_rate'],
               beta_1=config['beta_1'])

    d.trainable = True
    if config['multi_gpu']:
        d = multi_gpu_model(d)
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)

    d.trainable = False

    dcgan = Sequential([g, d])
    opt = Adam(lr=config['learning_rate'], beta_1=config['beta_1'])
    if config['multi_gpu']:
        dcgan = multi_gpu_model(dcgan)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['mae'],
                  optimizer=opt)

    return dcgan, d, g


if __name__ == '__main__':
    generator()
    discriminator()
