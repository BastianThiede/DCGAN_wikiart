import numpy as np
import pickle
import yaml
import math
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
import shutil
import multiprocessing


def load_image(image_path, input_height=256, input_width=256):
    try:
        img = cv2.imread(image_path)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(new_img, (input_height, input_width))
        scaled_img = scale(np.array(resized_img))
        return scaled_img
    except Exception:
        return None


def scale(img, reverse=False):
    if reverse:
        return (img * 255)
    else:
        return (img / 255.)


def combine_images_rgb(generated_images):
    total, width, height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    combined_image = np.zeros((height * rows, width * cols, 3),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[
            width * i:width * (i + 1), height * j:height * (j + 1), :
        ] = image[:, :, :]

    return combined_image


def get_config_path():
    dir_path = get_default_path()
    config_path = os.path.join(dir_path, 'configs')
    return config_path


def get_default_path():
    full_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(full_path)
    return dir_path


def load_config(config_file='default.yaml'):
    conf_path = get_config_path()
    print(conf_path, config_file)
    full_path = os.path.join(conf_path, config_file)
    with open(full_path) as f:
        config_yaml = yaml.load(f)

    return config_yaml


def load_data(path=None):
    if path is None:
        default_path = get_default_path()
        path = os.path.join(default_path, 'sample_data')
    search_path = os.path.join(path, '**/*.jpg')
    print('Searching at: {}'.format(search_path))
    paths = glob(search_path)[:5000]
    fpath = 'memmap_numpy_images.dat'
    data = np.memmap(fpath, dtype='float32', mode='w+',
                     shape=(len(paths), 256, 256, 3))
    pool = multiprocessing.Pool()
    counter = 0
    for img in tqdm(pool.imap_unordered(load_image, paths), total=len(paths)):
        if img is not None:
            data[counter, :, :, :] = img
            counter += 1
    return data


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_data_pkl(path=None):
    if path is None:
        default_path = get_default_path()
        path = os.path.join(default_path, 'sample_data')
    search_path = os.path.join(path, '**/*.pkl')
    print('Searching at: {}'.format(search_path))
    data = list()
    for pkl_path in tqdm(glob(search_path)):
        img = load_pkl(pkl_path)
        data.append(img)

    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 3)
    return np.array(data)


def get_image_paths(path=None):
    if path is None:
        default_path = get_default_path()
        path = os.path.join(default_path, 'sample_data')
    search_path = os.path.join(path, '**/*.jpg')
    all_image_paths = glob(search_path)
    return all_image_paths


def load_image_batch(paths, batch_size, batch_idx):
    batch = list()
    for path in paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]:
        try:
            img = load_image(path)
            batch.append(img)
        except Exception:
            pass

    while len(batch) < batch_size:
        try:
            img = load_image(random.choice(paths))
            batch.append(img)
        except Exception:
            pass

    return np.array(batch)


def save_images(images, save_name):
    combined = combine_images_rgb(scale(images, reverse=True))
    plt.imshow(combined.astype(np.uint8))
    plt.savefig(save_name)


def display_images(images):
    combined = combine_images_rgb(scale(images, reverse=True))
    plt.imshow(combined.astype(np.uint8))
    plt.show()


def get_gan_paths(save_dir):
    generator_path = os.path.join(save_dir, 'generator.h5')
    discriminator_path = os.path.join(save_dir, 'discriminator.h5')
    dcgan_path = os.path.join(save_dir, 'dcgan.h5')
    return dcgan_path, discriminator_path, generator_path


def sort_raw_files(unsorted_fpath):
    genre_folder = 'genre_links'
    new_path = '/tmp/wikiart_sorted'
    os.mkdir(new_path)
    for filename in os.listdir(genre_folder):
        genre_name = filename.split('_')[-1].replace('.txt', '')
        genre_path = os.path.join(new_path, genre_name)
        os.mkdir(genre_path)
        print('Ordering: {} to: {}'.format(genre_name, genre_path))
        with open(os.path.join(genre_folder, filename)) as f:
            files_to_move = f.read().split('\n')
            filenames = [x.split('/')[-1] for x in files_to_move[:-1]]
        for fname in tqdm(filenames):
            try:
                shutil.copyfile(os.path.join(unsorted_fpath, fname),
                                os.path.join(genre_path, fname))
            except FileNotFoundError:
                print('Skipping: {}'.format(fname))


def zero():
    return np.random.uniform(0.0, 0.01, size=[1])


def one():
    return np.random.uniform(0.99, 1.0, size=[1])


def noise(n):
    return np.random.uniform(-1.0, 1.0, size=[n, 4096])


if __name__ == '__main__':
    print(load_config())
    data = load_data()
    sample = data[0]
    print(sample.shape)
    display_images(data)
