from utils import (get_image_paths, load_data, display_images,
                   load_config, save_images, get_gan_paths, zero, one, noise)
from model import build_gan
import numpy as np
from time import time
import argparse
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


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

    z_pred = np.array([np.random.normal(-1, 1, 100) for _ in range(100)])
    y_g = [1] * batch_size
    y_d_true = [1] * batch_size
    y_d_gen = [0] * batch_size
    for epoch in range(epochs):
        start = time()
        for index in tqdm(range(num_batches)):
            X_train = load_image_batch(paths, batch_size, index)
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

    z_pred = noise(100)
    d_loss_fake_data = list()
    d_loss_real_data = list()
    g_loss_data = list()
    for epoch in range(epochs):
        y_g = np.array([zero() for _ in range(batch_size)])
        y_d_gen = np.array([one() for _ in range(int(batch_size / 2))])
        y_d_true = np.array([zero() for _ in range(int(batch_size / 2))])

        start = time()
        batches = list(range(num_batches))
        random.shuffle(batches)

        accuracy_real = list()
        accuracy_fake = list()

        for index in tqdm(batches):
            d_loss_fake, d_loss_real, g_loss = train_batch(X_train,
                                                           int(batch_size / 2),
                                                           dcgan,
                                                           discriminator,
                                                           generator, index,
                                                           y_d_gen, y_d_true,
                                                           y_g)
            d_loss_fake_data.append(d_loss_fake[0])
            d_loss_real_data.append(d_loss_real[0])
            g_loss_data.append(g_loss[0])

            accuracy_fake.append(d_loss_fake[1])
            accuracy_real.append(d_loss_real[1])

        end = time() - start
        mean_acc_real = np.mean(accuracy_real)
        mean_acc_fake = np.mean(accuracy_fake)

        # save generated images
        print('D-loss-real: {}, D-loss-fake: {}, '
              'G-loss: {}, epoch: {}, time: {}\n'
              'D-loss-real-mean: {}, D-loss-fake-mean: {}'.format(
                  d_loss_real, d_loss_fake, g_loss, epoch, end, mean_acc_real,
                  mean_acc_fake))

        if epoch % 5 == 0:
            X_d_true = X_train[index * batch_size:(index + 1) * batch_size]
            X_g = noise(batch_size)
            X_d_gen = generator.predict(X_g, verbose=0)

            disc_preds_true = discriminator.predict(X_d_true)
            disc_preds_fake = discriminator.predict(X_d_gen)
            print(y_d_true.shape)
            print()
            print(Counter(np.round(disc_preds_true[:, 0])), 'True_pred_count')
            print(np.mean(disc_preds_true), 'Mean_preds_true')
            print(np.std(disc_preds_true), 'Std_preds_true')
            print(np.mean(y_d_true), 'Mean_preds_true_labels')
            print(Counter(np.round(y_d_true[:, 0])), 'Counter_true_labels')
            print('-' * 72)
            print(Counter(np.round(disc_preds_fake[:, 0])), 'Fake_pred_count')
            print(np.mean(disc_preds_fake), 'Mean_preds_fake')
            print(np.std(disc_preds_fake), 'Std_preds_fake')
            print(np.mean(y_d_gen), 'Mean_preds_fake_labels')
            print(Counter(np.round(y_d_gen[:, 0])), 'Counter_fake_labels')
            print()

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
    X_g = noise(batch_size)
    X_d_gen = generator.predict(X_g, verbose=0)
    # train discriminator
    d_loss_real = discriminator.train_on_batch(X_d_true, y_d_true)
    d_loss_fake = discriminator.train_on_batch(X_d_gen, y_d_gen)
    # train generator
    X_g_full = noise(batch_size * 2)
    print(X_g_full.shape, len(y_g))
    g_loss = dcgan.train_on_batch(X_g_full, y_g)
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
