import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os, glob
import numpy as np
from PIL import Image
from skimage import measure
import scipy.ndimage.morphology as morphology
from data.data_utils import *


def get_centers_from_mask_XXX(x):
    """
    Args:
        x: input mask

    Returns:
        return coordinates
    """
    y = measure.label(x)
    coordinates = []
    for i in range(y.max()):
        coordinates.append((int(np.nonzero(y == i)[0].mean()),
                            int(np.nonzero(y == i)[1].mean())))
    return coordinates


def get_roi_from_mask(x, dis=100):
    """
    Args:
        x: input mask
        dis: maximum distance to a foreround pixle

    Returns: (x0, y0, x1, y1) coordinates of the ROI
    """
    y = (morphology.distance_transform_edt(1 - (x == x.max())) <= dis)
    return np.nonzero(y.sum(1))[0][0], np.nonzero(y.sum(0))[0][0],\
           np.nonzero(y.sum(1))[0][-1], np.nonzero(y.sum(0))[0][-1]


def get_bounding_boxes(x):
    boxes_all = []
    for i in range(x.shape[2]):
        blobs = measure.label(x[:, :, i])

        boxes = []
        for i in range(1, blobs.max() + 1):
            x0 = np.nonzero(blobs == i)[0].min()
            x1 = np.nonzero(blobs == i)[0].max()
            y0 = np.nonzero(blobs == i)[1].min()
            y1 = np.nonzero(blobs == i)[1].max()
            boxes.append((x0, y0, x1, y1))

        boxes_all.append(boxes)

    return boxes_all


def get_random_patches(img, mask, size=64, n=100):
    img_patches = []
    mask_patches = []
    for i in range(n):
        coor = (np.random.randint(img.shape[0]-size), np.random.randint(img.shape[1]-size))
        img_patches.append(np.expand_dims(img[coor[0]:coor[0] + size, coor[1]:coor[1] + size], 2))
        mask_patches.append(np.expand_dims(mask[coor[0]:coor[0] + size, coor[1]:coor[1] + size], 2))

    img_patches = np.concatenate(img_patches, 2)
    mask_patches = np.concatenate(mask_patches, 2)

    return img_patches, mask_patches


def draw_boxes_on(x, boxes):
    x0 = x/x.max()
    for _, b in enumerate(boxes):
        x0[b[0]:b[2] + 1, b[1]] = 0.5
        x0[b[0]:b[2] + 1, b[3]] = 0.5
        x0[b[0], b[1]:b[3] + 1] = 0.5
        x0[b[2], b[1]:b[3] + 1] = 0.5

    imagesc(x0)


dir_data = 'data/mouse_pathology/'
x = np.array(Image.open(dir_data + 'example_img.tiff'))
y = np.array(Image.open(dir_data + 'example_mask.tiff'))

# rough cropping
c = get_roi_from_mask(y)
x = x[c[0]:c[2], c[1]:c[3]]
y = y[c[0]:c[2], c[1]:c[3]]

# random patches
ip, mp = get_random_patches(x, y)
bp = get_bounding_boxes(mp)