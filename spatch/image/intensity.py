"""
Created on 29/04/13

@author: zw606
"""
from __future__ import division
import glob
from os.path import basename, isdir, join

import numpy
from mask import get_min_max_mask
from spatch.utilities.io import open_image, get_affine, save_3d_data
from spatch.utilities.paralleling import multi_process_list


__all__ = ["rescale_data", "rescale_file", "rescale_many_files", "threshold_image", "rescale_folder"]


def rescale_data(imageData, rescaleMin=0, rescaleMax=100, minDataValue=None, maxDataValue=None, maskData=None,
                 maskedRegionOnly=True):
    if rescaleMax - rescaleMin < 0:
        raise Exception("Incorrect rescaling values!")
    maskData = get_min_max_mask(imageData, minValue=minDataValue, maxValue=maxDataValue, maskData=maskData)
    if maskData is not None:
        imageMin = imageData[maskData].min()
        imageMax = imageData[maskData].max()
    else:
        imageMin = imageData.min()
        imageMax = imageData.max()

    if imageMin == rescaleMin and imageMax == rescaleMax:
        # no need to do anything
        return imageData

    scalingRange = rescaleMax - rescaleMin
    imageRange = imageMax - imageMin

    if maskData is not None and maskedRegionOnly:
        imageData[maskData] -= (imageMin - rescaleMin)
        imageData[maskData] *= (scalingRange / imageRange)
    else:
        imageData -= (imageMin - rescaleMin)
        imageData *= (scalingRange / imageRange)

    return imageData


def rescale_file(imageFile, savePath, rescaleMin=0, rescaleMax=100, minDataValue=None, maxDataValue=None, padding=None,
                 maskData=None, dtype=numpy.float32):
    imageData = open_image(imageFile)
    imageData = numpy.asarray(imageData, dtype=dtype)
    if padding is not None:
        if maskData is None:
            maskData = imageData != padding
        else:
            maskData = numpy.logical_and(maskData, padding)

    imageData = rescale_data(imageData, rescaleMin=rescaleMin, rescaleMax=rescaleMax, minDataValue=minDataValue,
                             maxDataValue=maxDataValue, maskData=maskData)
    if isdir(savePath):
        savePath = join(savePath, basename(imageFile))
    save_3d_data(imageData, get_affine(imageFile), savePath)


def rescale_worker(imageFiles, savePath, rescaleMin, rescaleMax, minDataValue, maxDataValue, padding,
                   maskData, queueOut):
    for f in imageFiles:
        rescale_file(f, savePath, rescaleMin=rescaleMin, rescaleMax=rescaleMax, minDataValue=minDataValue,
                     maxDataValue=maxDataValue, padding=padding, maskData=maskData)
        queueOut.put(1)


def rescale_many_files(filesList, savePath, rescaleMin=0, rescaleMax=100, minDataValue=None, maxDataValue=None,
                       padding=None, maskData=None, numProcessors=8):
    multi_process_list(filesList, rescale_worker, numProcessors, savePath, rescaleMin, rescaleMax, minDataValue,
                       maxDataValue, padding, maskData)


def rescale_folder(dataFolder, saveFolder, rescaleMin=0, rescaleMax=100, minDataValue=None,
                   maxDataValue=None, padding=None, maskData=None, numProcessors=8, fileName="*.nii.gz"):
    files = glob.glob(dataFolder + fileName)
    rescale_many_files(files, saveFolder, rescaleMin=rescaleMin, rescaleMax=rescaleMax, minDataValue=minDataValue,
                       maxDataValue=maxDataValue, padding=padding, maskData=maskData, numProcessors=numProcessors)


def threshold_image(imageData, minValue=None, maxValue=None):
    if maxValue is not None:
        imageData[imageData > maxValue] = maxValue
    if minValue is not None:
        imageData -= minValue
        imageData[imageData < 0] = 0
    return imageData
