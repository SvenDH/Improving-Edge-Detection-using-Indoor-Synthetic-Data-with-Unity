import io
import numpy as np
import os
import sys
import pandas as pd
import time
import h5py
import torch
from torch import nn
from PIL import Image
from scipy import ndimage
from scipy import misc
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from skimage.io import imsave

from const import *

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    return grad_y, grad_x


def imgrad_yx(img):
    N, C, w, h = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y, grad_x), dim=1)


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def image_to_array(image):
    return np.reshape(list(image.getdata()), (image.size[1], image.size[0], len(image.getbands())))


def png_to_array(bytes):
    return image_to_array(bytes_to_PIL(bytes))


def bytes_to_PIL(bytes):
    return Image.open(io.BytesIO(bytes))


def load_images(path, index):
    images = {}
    for type in texture_types:
        image = Image.open(os.path.join(path, type + "-" + str(index) + ".png"))
        images[type] = image_to_array(image)
    return images


def load_positions(path):
    positions = pd.read_csv(path)
    return positions.values


def one_hot_encode(labels):
    result = np.eye(len(class_labels))[labels]
    return result


def encode_colors(img):
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    for i, rgb in enumerate(class_labels):
        result[(img == rgb).all(2)] = i
    return result


def decode_colors(labels):
    # labels.shape() == (:,:,len(class_labels))
    return class_labels[labels].astype(np.uint8)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def find_optimal_threshold(precision, recall, thresholds):
    A = np.linspace(0, 1, 100)
    B = 1 - A
    bestF = -1
    for i, th in enumerate(thresholds[1:]):
        R = recall[i] * A + recall[i - 1] * B
        P = precision[i] * A + precision[i - 1] * B
        T = thresholds[i] * A + thresholds[i - 1] * B
        F = (2 * P * R) / (P + R)
        k = np.argmax(F)
        if F[k] > bestF:
            bestT = T[k]
            bestR = R[k]
            bestP = P[k]
            bestF = F[k]

    return bestT, bestR, bestP, bestF


def nyud_split(filename):
    train_test = scipy.io.loadmat(filename)

    test_images = [int(x)-1 for x in train_test["testNdxs"]]
    train_images = [int(x)-1 for x in train_test["trainNdxs"]]

    return train_images, test_images
