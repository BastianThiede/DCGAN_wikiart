import numpy as np
import scipy.misc
import yaml
import math
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image(image_path, input_height=256, input_width=256):
    img = scipy.misc.imread(image_path).astype(np.float)
    resized_img = scipy.misc.imresize(img, [input_height, input_width])
    scaled_img = scale(np.array(resized_img))
    return scaled_img


def scale(img, reverse=False):
    if reverse:
        return (img * 127.5) + 127.5
    else:
        return (img - 127.5) / 127.5


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
    data = list()
    for image_path in tqdm(glob(search_path)):
        data.append(load_image(image_path))
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 3)
    return np.array(data)


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

if __name__ == '__main__':
    print(load_config())
    data = load_data()
    sample = data[0]
    print(sample.shape)
    display_images(data)
