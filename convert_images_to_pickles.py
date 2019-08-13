import pickle
from utils import get_image_paths, load_image
from tqdm import tqdm
import sys
import os


def convert_images(pickle_save_dir, image_dir):
    image_paths = get_image_paths(image_dir)
    for path in tqdm(image_paths):
        fname = '{}.pkl'.format(path.split('/')[-1].replace('.jpg', ''))
        try:
            img = load_image(path)
        except Exception:
            pass

        with open(os.path.join(os.path.join(pickle_save_dir, fname)),
                  'wb') as f:
            pickle.dump(img, f)


if __name__ == '__main__':
    convert_images(pickle_save_dir=sys.argv[1],
                   image_dir=sys.argv[2])
