from keras.initializers import RandomNormal
from keras.layers import (Conv2DTranspose, Conv2D, BatchNormalization, Reshape,
                          Dense, LeakyReLU, ReLU, GaussianNoise, Dropout,Activation,
                          AveragePooling2D, Flatten, UpSampling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from utils import load_config


def generator(input_dim=100,units=1024,activation='relu'):
    init = RandomNormal(stddev=0.02)

    # Generator network
    generator = Sequential()
    # FC: 2x2x512
    generator.add(Dense(4*4*1024,input_shape=(input_dim,),activation='sigmoid',kernel_initializer=init))
    generator.add(Reshape((4, 4, 1024)))

    # Conv 2: 8x8x128
    generator.add(Conv2DTranspose(512, kernel_size=5,kernel_initializer=init, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    # Conv 3: 16x16x64
    generator.add(Conv2DTranspose(256, kernel_size=5,kernel_initializer=init, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    generator.add(Conv2DTranspose(128, kernel_size=5,kernel_initializer=init, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    generator.add(
        Conv2DTranspose(64, kernel_size=5, strides=2,kernel_initializer=init, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    generator.add(
        Conv2DTranspose(32, kernel_size=5, strides=2,kernel_initializer=init, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(ReLU())

    generator.add(Conv2DTranspose(3, kernel_size=3,kernel_initializer=init, strides=2, padding='same',
                                  activation='tanh'))
    print(generator.summary())
    return generator

def discriminator(input_shape=(32, 32, 3),nb_filter=64):
    init = RandomNormal(stddev=0.02)

    discriminator = Sequential()

    # Conv 1: 16x16x64
    discriminator.add(GaussianNoise(0.2, input_shape=(256, 256, 3)))

    discriminator.add(Conv2D(16, kernel_size=5, strides=2, padding='same',
                            kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(AveragePooling2D())


    # Conv 2:
    discriminator.add(Conv2D(32, kernel_size=5, kernel_initializer=init, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(AveragePooling2D())

    discriminator.add(Conv2D(64, kernel_size=5, kernel_initializer=init, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))


    discriminator.add(Conv2D(128, kernel_size=5, kernel_initializer=init, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))



    # FC
    discriminator.add(Flatten())

    # Output
    discriminator.add(Dense(1, activation='sigmoid'))
    print(discriminator.summary())
    return discriminator
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
                  metrics=['accuracy'],
                  optimizer=opt)

    return dcgan, d, g


if __name__ == '__main__':
    generator()
    discriminator()
