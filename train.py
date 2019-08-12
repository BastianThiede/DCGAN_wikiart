from utils import load_data, load_config, save_images
from model import build_gan
import numpy as np
from time import time
import sys


def main(config_path='default.yaml'):
    config = load_config(config_path)
    batch_size = config['batch_size']
    epochs = config['epochs']
    X_train = load_data()
    dcgan, discriminator, generator = build_gan(config_path)

    num_batches = int(X_train.shape[0] / batch_size)

    print("-------------------")
    print("Total epoch:", config['epochs'], "Number of batches:", num_batches)
    print("-------------------")

    z_pred = np.array([np.random.normal(0, 0.5, 100) for _ in range(100)])
    y_g = [1] * batch_size
    y_d_true = [1] * batch_size
    y_d_gen = [0] * batch_size
    for epoch in range(epochs):
        start = time()
        for index in range(num_batches):
            d_loss_fake, d_loss_real, g_loss = train_batch(X_train, batch_size,
                                                           dcgan,
                                                           discriminator,
                                                           generator, index,
                                                           y_d_gen, y_d_true,
                                                           y_g)
        end = time() - start
        # save generated images
        print('D-loss-real: {}, D-loss-fake: {}, '
              'G-loss: {}, epoch: {}, time: {}'.format(d_loss_real,
                                                       d_loss_fake,
                                                       g_loss,
                                                       epoch,
                                                       end))

        if epoch % 10 == 0:
            images = generator.predict(z_pred)
            save_images(images, 'dcgan_keras_epoch_{}.png'.format(epoch))


def train_batch(X_train, batch_size, dcgan, discriminator, generator, index,
                y_d_gen, y_d_true, y_g):
    X_d_true = X_train[index * batch_size:(index + 1) * batch_size]
    X_g = np.array([np.random.normal(0, 0.5, 100) for _ in range(batch_size)])
    X_d_gen = generator.predict(X_g, verbose=0)
    # train discriminator
    d_loss_real = discriminator.train_on_batch(X_d_true, y_d_true)
    d_loss_fake = discriminator.train_on_batch(X_d_gen, y_d_gen)
    # train generator
    g_loss = dcgan.train_on_batch(X_g, y_g)
    return d_loss_fake, d_loss_real, g_loss


if __name__ == '__main__':
    main(sys.argv[1])
