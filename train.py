from utils import get_image_paths, load_data, load_config, save_images, get_gan_paths
from model import build_gan
import numpy as np
from time import time
import argparse
import random
import os
from tqdm import tqdm

def train_batches(config_path, save_dir, data_dir):
    config = load_config(config_path)
    batch_size = config['batch_size']
    dcgan_path, generator_path, discriminator_path = get_gan_paths(save_dir)

    dcgan, discriminator, generator = load_or_create_model(config_path,
                                                           dcgan_path,
                                                           discriminator_path,
                                                           generator_path)

    epochs = config['epochs']
    paths = get_image_paths(data_dir)
    num_batches = int(len(paths) / batch_size)

    print("-------------------")
    print("Total epoch:", config['epochs'], "Number of batches:", num_batches)
    print("-------------------")

    z_pred = np.array([np.random.normal(0, 0.5, 100) for _ in range(100)])
    y_g = [1] * batch_size
    y_d_true = [1] * batch_size
    y_d_gen = [0] * batch_size
    for epoch in range(epochs):
        start = time()
        for index in tqdm(range(num_batches)):
            X_train = load_image_batch(paths,batch_size,index)
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
            generator.save(generator_path)
            discriminator.save(discriminator_path)
            dcgan.save(dcgan_path)
            images = generator.predict(z_pred)
            save_images(images, 'dcgan_keras_epoch_{}.png'.format(epoch))

def main(config_path, save_dir, data_dir):
    config = load_config(config_path)
    batch_size = config['batch_size']
    dcgan_path, generator_path, discriminator_path = get_gan_paths(save_dir)

    dcgan, discriminator, generator = load_or_create_model(config_path,
                                                           dcgan_path,
                                                           discriminator_path,
                                                           generator_path)

    epochs = config['epochs']
    X_train = load_data(path=data_dir)
    num_batches = int(X_train.shape[0] / batch_size)

    print("-------------------")
    print("Total epoch:", config['epochs'], "Number of batches:", num_batches)
    print("-------------------")

    z_pred = np.array([np.random.normal(0, 0.5, 100) for _ in range(100)])
    y_d_true = [1] * batch_size
    for epoch in range(epochs):
        y_g = [(1 - random.randrange(0, 5) / 100.) for _ in range(batch_size)]
        y_d_gen = [random.randrange(0, 5) / 100. for _ in range(batch_size)]

        start = time()
        batches = list(range(num_batches))
        random.shuffle(batches)
        for index in tqdm(batches):
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
            generator.save(generator_path)
            discriminator.save(discriminator_path)
            dcgan.save(dcgan_path)
            images = generator.predict(z_pred)
            save_images(images, 'dcgan_keras_epoch_{}.png'.format(epoch))


def load_or_create_model(config_path, dcgan_path, discriminator_path,
                         generator_path):
    dcgan, discriminator, generator = build_gan(config_path)

    if (os.path.exists(dcgan_path) and os.path.exists(generator_path)
            and os.path.exists(discriminator_path)):
        dcgan.load_weights(dcgan_path)
        generator.load_weights(generator_path)
        discriminator.load_weights(discriminator_path)

    return dcgan, discriminator, generator


def train_batch(X_train, batch_size, dcgan, discriminator, generator, index,
                y_d_gen, y_d_true, y_g):
    X_d_true = X_train[index * batch_size:(index + 1) * batch_size]
    #X_d_true = X_d_true.view(dtype=np.float32, type=np.ndarray)
    X_g = np.array([np.random.normal(0, 0.5, 100) for _ in range(batch_size)])
    X_d_gen = generator.predict(X_g, verbose=0)
    # train discriminator
    d_loss_real = discriminator.train_on_batch(X_d_true, y_d_true)
    d_loss_fake = discriminator.train_on_batch(X_d_gen, y_d_gen)
    # train generator
    g_loss = dcgan.train_on_batch(X_g, y_g)
    return d_loss_fake, d_loss_real, g_loss


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", help="config filename",
                        default='test_config.yaml')
    parser.add_argument("--data_dir", help="data-dir", default=None)
    parser.add_argument("--save_dir", help="save dir", default='/tmp')
    return parser


if __name__ == '__main__':
    parser = build_argparse()
    args = parser.parse_args()
    main(config_path=args.config_filename,
         data_dir=args.data_dir,
         save_dir=args.save_dir)
