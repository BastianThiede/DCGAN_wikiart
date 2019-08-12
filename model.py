from keras.initializers import RandomNormal
from keras.layers import (ZeroPadding2D,
    Conv2DTranspose, Conv2D, BatchNormalization, Reshape, Dense, LeakyReLU,
    Flatten,UpSampling2D)
from keras.models import Sequential


def generator(input_dim=100, units=1024, activation='relu'):
    init = RandomNormal(stddev=0.02)

    # Generator network
    generator = Sequential()

    # FC: 2x2x512
    generator.add(
        Dense(4 * 4 * 1024, input_shape=(input_dim,), kernel_initializer=init)
    )
    generator.add(Reshape((4, 4, 1024)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(UpSampling2D())

    # # Conv 1: 4x4x256

    generator.add(
        Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')
    )
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    generator.add(
        Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 3: 16x16x64
    generator.add(
        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # Conv 4: 32x32x3
    generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same',
                                  activation='tanh'))
    print(generator.summary())
    return generator


def discriminator():
    init = RandomNormal(stddev=0.02)

    discriminator = Sequential()

    # Conv 1: 16x16x64
    discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding='same',
                             input_shape=(256, 256, 3), kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))

    # Conv 2:
    discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # Conv 3:
    discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # Conv 3:
    discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    # FC
    discriminator.add(Flatten())

    # Output
    discriminator.add(Dense(1, activation='sigmoid'))
    print(discriminator.summary())
    return discriminator


if __name__ == '__main__':
    generator()
    discriminator()