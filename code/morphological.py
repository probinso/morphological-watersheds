#!/usr/bin/env python3

from functools import reduce, partial
from itertools import combinations
from multiprocessing import Pool
from operator import or_
import os
import os.path as osp
import sys

import numpy as np
from scipy.misc import imread, imsave

from helpers import grayprep as prep, digest_paths, get_regions, \
    list_regions, lmap, ndi, label2rgb


def T(G, n, mask):
    return (G < n) & mask


def Cn_Mi(G, n, M):
    return T(G, n) & M


def catchments(G, n, Minima):
    return tuple(Cn_Mi(G, n, M) for M in Minima)


def B(G, n, Minima):
    pools = catchments(G, n, Minima)
    return np.sum(np.any(p) for p in pools)


def C(G, n, Minima):
    pools = catchments(G, n, Minima)
    return reduce(or_, pools, np.zeros_like(G))


def Q(G, n, mask):
    return get_regions(T(G, n, mask))


def itemize(it):
    for i, _ in enumerate(it):
        yield i


def minus_dams(q, regions, footprint=np.ones((3,3), dtype=np.bool), cores=7):

    union = lambda imgs: reduce(or_, imgs, np.zeros_like(q))
    dialate = partial(ndi.binary_dilation, structure=footprint, mask=q)

    overlap = lmap(np.zeros_like, regions)

    while True:
        dialation = lmap(dialate, regions)

        for a, b in combinations(itemize(dialation), 2):
            roi = dialation[a] & dialation[b]
            overlap[a] |= roi
            overlap[b] |= roi

        for key in itemize(dialation):
            regions[key] = dialation[key] & ~overlap[key]

        if np.all(q == (union(regions) | union(overlap))):

            dialation = lmap(dialate, regions)

            for a, b in combinations(itemize(dialation), 2):
                roi = dialation[a] & dialation[b]
                overlap[a] |= roi
                overlap[b] |= roi

            for key in itemize(dialation):
                regions[key] = dialation[key] & ~overlap[key]

            break

    return regions, union(overlap)


def watershed(Z, n, mask=None):

    retval = np.zeros_like(Z)
    if mask is None:
        mask = np.ones_like(Z, dtype='bool')

    for i in range(n):
        prior = np.copy(retval)

        regions, total = Q(Z, i, mask)
        for q in (regions == i for i in range(1, total+1)):
            intersection = q & (prior > 0)
            collections, count = get_regions(intersection)

            if count == 0:
                retval[np.where(q)] = np.max(retval) + 1
            elif count == 1:
                idx = np.where(q)
                retval[idx] = np.max(retval[idx])
            else: # count > 1
                grows, e = minus_dams(q, list_regions(intersection))
                #print(Z.dtype)
                Z[np.where(e)] = np.iinfo(Z.dtype).max
                for g in (g for g in grows if np.any(g)):
                    idx = np.where(g)
                    retval[idx] = np.max(retval[idx])

        yield retval


def interface(inpath, outpath, cap):

    signature = digest_paths(inpath)
    dst = osp.join(outpath, signature)
    os.makedirs(dst, exist_ok=True)

    raw = imread(inpath)
    processed = prep(raw).astype(np.uint32)
    data = watershed(processed, cap, np.zeros_like(processed, dtype='bool'))

    print(np.unique(data[-1]))
    for i, regions in enumerate(data):
        result = np.copy(raw)
        result = label2rgb(regions, image=result, bg_label=0)
        imsave(osp.join(dst, f'img-{str.zfill(str(i),3)}.png'), result)

    np.save(osp.join(dst, 'raw.npy'), raw)
    np.save(osp.join(dst, 'reg.npy'), data[-1])

def cli_interface():
    """
    by convention it is helpful to have a wrapper_cli method that interfaces
    from commandline to function space.
    """
    try:
        inpath, outpath, cap = sys.argv[1:]
    except:
        print("usage: {}  <inpath>  <dest>  <cap>".format(sys.argv[0]))
        sys.exit(1)
    interface(inpath, outpath, int(cap))


if __name__ == '__main__':
    cli_interface()
