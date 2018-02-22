#!/usr/bin/env python3

from functools import reduce, partial
import hashlib

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.filters import minimum_filter, gaussian_filter

from skimage.color import label2rgb
from skimage.util import invert

def compose(*funcs):
    return reduce(lambda g, h: lambda x: g(h(x)),
                  funcs, lambda x: x)

get_regions = ndi.label

def local_minima_regions(V, shape=(30,30)):
    footprint = np.ones(shape)
    target = minimum_filter(V, footprint=footprint) != V
    return target == 0

get_minima_regions = compose(get_regions, local_minima_regions)

lmap = lambda f, l: list(map(f, l))


def list_regions(V):
    regions, count = get_regions(V)
    return list(regions == target for target in range(1, count+1))


def digest_paths(*paths):
    """
      takes in a list containing paths to files
      returns digest SHA hash to be used as identifier over the whole list
    """
    HASH = hashlib.sha1()

    for filename in paths:
        buf = filename
        HASH.update(buf.encode('utf-8'))

        with open(filename, 'rb') as fd:
            while True:
                # Read files as little chunks to prevent large ram usage
                buf = fd.read(4096)
                if not buf : break
                HASH.update(buf)

    return HASH.hexdigest()


def channel_op(img, op=lambda x: x):
    x, y, c = img.shape
    return np.dstack(map(op, (img[:, :, i] for i in range(c))))

makegray = partial(np.mean, axis=2)
takemin = partial(np.max, axis=2)

smooth = lambda s: lambda img: gaussian_filter(img, s)
invert = invert # lambda img: np.max(img) - img

chsmooth = partial(channel_op, op=smooth)

minprep = compose(invert, takemin, chsmooth)
grayprep = compose(invert, smooth(3), makegray)
