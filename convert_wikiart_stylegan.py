import sys
import os
from glob import glob
import cv2
import os
from utils import get_default_path
import multiprocessing
from tqdm import tqdm
from functools import partial

def main(path=None):
    if path is None:
        default_path = get_default_path()
        path = os.path.join(default_path, 'sample_data')
    search_path = os.path.join(path, '**/*.jpg')
    print('Searching at: {}'.format(search_path))
    paths = glob(search_path)
    resolution = 256
    image_folder = '/tmp/processed_images_{}'.format(resolution)
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    proc_partial = partial(process_image,
                           resolution=resolution,
                           image_folder=image_folder)
    counter = 0
    with multiprocessing.Pool() as pool:
        for val in tqdm(pool.imap_unordered(proc_partial, paths),
                        total=len(paths)):
            if val == True:
                counter +=1
    print('Processed {} images'.format(counter))




def process_image(image_path,resolution=256,image_folder='/tmp'):
    try:
        fname = image_path.split('/')[-1]
        img = cv2.imread(image_path)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(new_img, (resolution, resolution))
        new_path = os.path.join(image_folder,fname)
        cv2.imwrite(new_path,cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        return True
    except Exception:
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None
    main(filepath)