"""
Created on 23 Apr 2012

@author: Zehan Wang
"""

from __future__ import division

import numpy
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

__all__ = ["resample_data_to_shape", "interpolate_to_shape",
           "crop_image", "zero_out_boundary", "image_boundary_expand"]


def resample_data_to_shape(data, targetShape, interpolationType, isLabels=False):
    if data.shape == targetShape:
        return data
    if not isLabels and requires_gaussian(data.shape, targetShape):
        sigmas = get_sigmas(data.shape, targetShape)
        data = gaussian_filter(data, tuple(sigmas))
    data = interpolate_to_shape(data, targetShape, interpolationType=interpolationType)
    return data


def requires_gaussian(dataShape, targetShape):
    for i in xrange(len(dataShape)):
        if dataShape[i] > targetShape[i]:
            return True
    return False


def get_sigmas(dataShape, targetShape):
    multipliers = numpy.asarray(dataShape, dtype=numpy.float64) / numpy.asarray(targetShape, dtype=numpy.float64)
    for i in xrange(len(multipliers)):
        if multipliers[i] > 1:
            multipliers[i] /= 2
        else:
            multipliers[i] = 0

    return multipliers


def interpolate_to_shape(data, targetShape, interpolationType="NN"):
    """

    @param data: data which needs to be interpolated to given shape
    @param targetShape: the shape (tuple) that data should be interpolated to
    @param interpolationType: choices of <"NN" (default), "LINEAR", "QUADRATIC", "CUBIC">
    """
    if data.shape != targetShape:
        order = 0
        if interpolationType.lower() == "linear":
            order = 1
        elif interpolationType.lower() == "quadratic":
            order = 2
        elif interpolationType.lower() in ["cubic", "bspline"]:
            order = 3
        if order == 0:
            # nearest neighbour - most likely used for labels
            data = resize(data, targetShape, order=order, mode="constant", cval=data.min())
        else:
            data = numpy.float64(data)
            data = resize(data, targetShape, order=order, mode="nearest")
    return data


def get_scale_multipliers(inputShape, targetShape):
    multipliers = numpy.asarray(targetShape) / numpy.asarray(inputShape)
    endShape = tuple([int(ii * jj) for ii, jj in zip(inputShape, multipliers)])
    for i in range(len(endShape)):
        if endShape[i] != targetShape[i]:
            multipliers[i] = (targetShape[i] + 0.45) / inputShape[i]
    return multipliers


def crop_image(image, mask, padding):
    if padding is not None:
        tempMask = mask * (image != padding)
        paddedArea = numpy.logical_or((image == padding), (mask == 0))
        image *= tempMask

        image += (paddedArea * padding)

        return image
    else:
        return image * mask


def zero_out_boundary(imageData, boundarySize, replacementValue=0):
    if isinstance(boundarySize, int):
        if boundarySize < 1:
            return imageData
        boundaryX = boundarySize
        boundaryY = boundarySize
        boundaryZ = boundarySize
    else:
        boundarySize = tuple(boundarySize)
        if len(boundarySize) == 1:
            if boundarySize[0] < 1:
                return imageData
            boundaryX = boundarySize[0]
            boundaryY = boundarySize[0]
            boundaryZ = boundarySize[0]
        else:
            boundaryX = boundarySize[0]
            boundaryY = boundarySize[1]
            boundaryZ = boundarySize[2]

    imageData[:boundaryX, :, :] = replacementValue
    imageData[-boundaryX:, :, :] = replacementValue
    if boundaryY > 0:
        imageData[:, :boundaryY, :] = replacementValue
        imageData[:, -boundaryY:, :] = replacementValue
    if boundaryZ > 0:
        imageData[:, :, :boundaryZ] = replacementValue
        imageData[:, :, -boundaryZ:] = replacementValue

    return imageData


def image_boundary_expand(imageData, useGradient=True, minValue=None, is2D=False):
    """
        expands the image boundaries based on the gradient at the edges
        returns an image that is 2 voxels larger in all axis
    """
    results = numpy.zeros(tuple(numpy.asarray(imageData.shape) + 2), imageData.dtype)
    results[1:-1, 1:-1, 1:-1] = imageData
    if is2D:
        results = numpy.zeros(tuple(numpy.asarray(imageData.shape) + numpy.asarray([2, 2, 0])), imageData.dtype)
        results[1:-1, 1:-1] = imageData

    if useGradient:
        results[0, :, :] = results[1, :, :] - (
            (results[3, :, :] - results[1, :, :]) / 2 + (results[2, :, :] - results[1, :, :])) / 2
        results[:, 0, :] = results[:, 1, :] - (
            (results[:, 3, :] - results[:, 1, :]) / 2 + (results[:, 2, :] - results[:, 1, :])) / 2

        results[-1, :, :] = results[-2, :, :] - (
            (results[-4, :, :] - results[-2, :, :]) / 2 + (results[-3, :, :] - results[-2, :, :])) / 2
        results[:, -1, :] = results[:, -2, :] - (
            (results[:, -4, :] - results[:, -2, :]) / 2 + (results[:, -3, :] - results[:, -2, :])) / 2
        if not is2D:
            results[:, :, 0] = results[:, :, 1] - (
                (results[:, :, 3] - results[:, :, 1]) / 2 + (results[:, :, 2] - results[:, :, 1])) / 2
            results[:, :, -1] = results[:, :, -2] - (
                (results[:, :, -4] - results[:, :, -2]) / 2 + (results[:, :, -3] - results[:, :, -2])) / 2
    if minValue is not None:
        results[results < minValue] = 0
    return results