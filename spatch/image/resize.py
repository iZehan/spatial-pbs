"""
Created on 11 May 2012

@author: zw606
"""

import glob
from utilities import io
import os
import numpy
import mask


def crop_image(image, mask, padding):
    if padding is not None:
        tempMask = mask * (image != padding)
        paddedArea = numpy.logical_or((image == padding), (mask == 0))
        image *= tempMask

        image += (paddedArea * padding)

        return image
    else:
        return image * mask


def crop_images(dataPath, maskFile, savePath, padding, maskDilation=20):
    dataFiles = glob.glob(dataPath + "*.nii.gz")
    imMask = io.open_image(maskFile)
    if maskDilation > 0:
        print "Dilating mask by", maskDilation
        imMask = mask.dilate_mask(imMask, maskDilation)
    for x in dataFiles:
        print "Cropping", x
        data = io.open_image(x)
        affine = io.get_affine(x)
        newData = crop_image(data, imMask, padding)
        io.save_3d_data(newData, affine, savePath + os.path.basename(x))

    print "Cropped", len(dataFiles), "Images"


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
        expands the image boundaries based on the Hog at the edges
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


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option("--dp", dest="dataPath",
                      help="Set the data path")
    parser.add_option("--mf", dest="maskFile",
                      help="Set the labels path")
    parser.add_option("--sp", dest="savePath",
                      help="Set the labels path")
    parser.add_option("--pad", dest="padding", default=None, type="int",
                      help="Set padding value")
    parser.add_option("--dilation", dest="dilation", default=0, type="int",
                      help="Set mask dilation for area to be cropped")

    (options, argPos) = parser.parse_args()

    crop_images(options.dataPath, options.maskFile, options.savePath, options.padding, options.dilation)

    print "Finished!"