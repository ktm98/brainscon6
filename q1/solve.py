import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

from tqdm.auto import tqdm
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import glob

import warnings 
warnings.filterwarnings('ignore')

# cv2
# ref: https://note.nkmk.me/python-opencv-hconcat-vconcat-np-tile/
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

# PIL
# ref
def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)


# cv2
def concat_images(files):
    images = [cv2.imread(img) for img in files]
    images2d = np.array(images)
    n = 14
    assert 252%n == 0
    images2d = images2d.reshape(n, 252//n, 256, 256, 3)
    img_tile = concat_tile(images2d)
    return img_tile

# PIL
def concat_images_pil(files):
    images = [Image.open(img) for img in files]
    images2d = np.array(images, dtype=np.object)
    n = 14
    assert 252%n == 0
    images2d = images2d.reshape(n, 252//n)
    img_tile = get_concat_tile_resize(images2d)
    return img_tile


if __name__ == '__main__':
    IMAGE_DIR = '../input/SDNET2018/W/'
    OUTPUT_DIR = './output/'
    image_files = []
    for i in range(252):
        image_files.append(glob.glob(IMAGE_DIR+f'*/7069-{i+1}.jpg')[0])

    image = concat_images_pil(image_files)

    image.save(OUTPUT_DIR + 'sample_submission.png')

    # cv2.imwrite(OUTPUT_DIR + 'sample_submission.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION ), 7])

