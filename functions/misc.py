# coding: utf-8
import numpy as np
from scipy import ndimage as ndi

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

def rewrite_data(filename, database):
    '''
    Rewrite database names in the prototxt file
    '''
    net = caffe_pb2.NetParameter()
    with open(filename) as f:
        s = f.read()
        txtf.Merge(s, net)

    layerNames = [l.name for l in net.layer]
    print(layerNames)
    idx = layerNames.index('data')
    data = net.layer[idx]
    data.data_param.source = database

def random_crop(image, crop_dims, crop_num):
    """
    Crop images randomly

    Parameters
    ----------
    image: (H x W x K) ndarray
    crop_dims: (height, width) tuple for the crops
    crop_num: number of randomly cropping

    Returns
    -------
    crops: (N x H x W x K) ndarray of crops for number of crop_num (N).
    """
    # Dimensions
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)

    # Coordinate of top-left corner
    limit_y = int(im_shape[0] - crop_dims[0])
    limit_x = int(im_shape[1] - crop_dims[1])
    # 0 ~ (limit-1) の整数で(10, 2)のarrayを作成
    im_toplefts_x = np.random.randint(0, limit_x, (10, 1))
    im_toplefts_y = np.random.randint(0, limit_y, (10, 1))
    im_toplefts = np.concatenate([im_toplefts_y, im_toplefts_x], axis=1)

    # Make crop coordinates
    crop_ix = np.concatenate([
         im_toplefts,
         im_toplefts + np.tile(crop_dims, (10, 1))
    ], axis=1)
    crop_ix = crop_ix.astype(int)

    crops = np.empty((crop_num, crop_dims[0], crop_dims[1], im_shape[-1]), dtype=image.dtype)
    for ix in range(crop_num):
        crops[ix] = image[crop_ix[ix, 0]:crop_ix[ix, 2], crop_ix[ix, 1]:crop_ix[ix, 3], :]

    return crops

def center_crop(image, crop_dims):
    """
    Crop images into the center

    Parameters
    ----------
    image : (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    # coordinates of center cropping
    crop_ix = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crop_ix = crop_ix.astype(int)

    crop = image[crop_ix[0, 0]:crop_ix[0, 2], crop_ix[0, 1]:crop_ix[0, 3], :]

    return crop

def RotateImage(images):
    """
    Rotate images with 90, 180, 270 degrees and their mirrored versions

    Parameters
    ----------
    images: target RGB images (N x C x H x W)

    Returns
    -------
    rotate_images: rotated & mirrored images (8N x C x H x W)
    """
    images = images.transpose(0, 2, 3, 1)
    im_shape = images.shape
    rotate_images = np.empty((0, im_shape[1], im_shape[2], im_shape[3]), np.float32)
    image_num = 0
    for image in images:
        for i in range(4):
            if i == 0:
                image_rot = image
            else:
                image_rot = ndi.rotate(image_rot, 90)
            rotate_images = np.append(rotate_images, [image_rot], axis=0)
            image_rot_ud = np.flipud(image_rot)
            rotate_images = np.append(rotate_images, [image_rot_ud], axis=0)
            image_num += 1

    rotate_images = rotate_images.transpose(0, 3, 1, 2)
    return rotate_images

def downsampling(images, levels):
    div = 2**(levels)
    images_t = np.empty((0, int(images.shape[1])/div, int(images.shape[2])/div), np.float32)
    for image in images:
        tmp = image
        for level in range(levels):
            tmp = (tmp[0::2, 0::2] + tmp[1::2, 0::2] + tmp[0::2, 1::2] + tmp[1::2, 1::2]) / 4
        image_t = tmp
        images_t = np.append(images_t, [image_t], axis=0)
    return images_t

def gram_matrix(activations):
    '''
    Gives the gram matrix for feature map activations in caffe format with batchsize 1. Normalises by spatial dimensions.

    :param activations: feature map activations to compute gram matrix from
    :return: normalised gram matrix
    '''
    N = activations.shape[1]
    F = activations.reshape(N,-1)
    M = F.shape[1]
    G = np.dot(F,F.T) / M
    return G, F

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True, sqrt_bias=10., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    --------
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other parameters) corresponds to the preprocessing used in [1].
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the per-pixel mean across examples.
    # So it is perfectly fine to subtract this without worrying about whether the current object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X
