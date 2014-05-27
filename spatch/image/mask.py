"""
Created on 22 Mar 2012

@author: zw606
"""

from scipy.ndimage import morphology
import numpy

from spatch.utilities.misc import auto_non_background_labels

__all__ = ["get_data_mask", "get_min_max_mask", "get_bounding_box", "get_boundary_mask", "get_boundary_masks",
           "get_label_masks", "dilate_mask", "erode_mask", "open_mask", "close_mask", "union_masks"]


def get_data_mask(imData, labels, boundarySize=None, specificLabels=None, roiMask=None, minValue=None, maxValue=None):
    if roiMask is None and boundarySize is None and minValue is None and maxValue is None:
        return None
    if roiMask is None:
        if imData is not None:
            maskData = numpy.ones(imData.shape, numpy.bool)
        else:
            maskData = numpy.ones(labels.shape, numpy.bool)
    else:
        maskData = roiMask
    if minValue is not None:
        maskData = numpy.logical_and(imData > minValue, maskData)
    if maxValue is not None:
        maskData = numpy.logical_and(imData < maxValue, maskData)
    if boundarySize:
        maskData = numpy.logical_and(get_boundary_mask(labels, boundarySize, specificLabels=specificLabels), maskData)
    return maskData


def get_min_max_mask(image, minValue=None, maxValue=None, maskData=None):
    if minValue is not None:
        minValueMask = image > minValue
        if maskData is not None:
            maskData = numpy.logical_and(minValueMask, maskData)
        else:
            maskData = minValueMask
    if maxValue is not None:
        maxValueMask = image < maxValue
        if maskData is not None:
            maskData = numpy.logical_and(maxValueMask, maskData)
        else:
            maskData = maxValueMask
    return maskData


def get_bounding_box(maskData):
    temp = numpy.argwhere(maskData)
    return temp.min(0), temp.max(0) + 1


def get_boundary_mask(labelledData, boundaryDilation, specificLabels=None, cubic=False, is2D=False):
    return union_masks(get_boundary_masks(labelledData, boundaryDilation, specificLabels=specificLabels,
                                          isFullSized=cubic, is2D=is2D))


def get_boundary_masks(labelledData, boundaryDilation, specificLabels=None, isFullSized=False, is2D=False):
    """ returns list of boundary masks for specified labels"""
    if specificLabels is None:
        specificLabels = auto_non_background_labels(labelledData)

    return [dilate_mask(labelledData == label, boundaryDilation, isFullSized=isFullSized, is2D=is2D)
            - erode_mask(labelledData == label, boundaryDilation, isFullSized=isFullSized, is2D=is2D)
            for label in specificLabels]


def get_label_masks(labelledData, specificLabels=None, cubic=False):
    """ returns list of boundary masks for specified labels"""
    if specificLabels is None:
        specificLabels = numpy.unique(labelledData)

    return [labelledData == label for label in specificLabels]


def fill_holes(maskData):
    return morphology.binary_fill_holes(maskData)


def fill_holes_by_value(data, fill_holes_by_value):
    zeroLabels = (data == 0)
    for i in fill_holes_by_value:
        data += numpy.int16(fill_holes(data == i) * zeroLabels) * i
    return data


def get_structuring_element(isFullSized=False, is2D=False):
    structuringElement = None
    if is2D:
        structuringElement = numpy.ones((3, 3, 1), numpy.bool)
        if not isFullSized:
            structuringElement[0, 0] = 0
            structuringElement[0, 2] = 0
            structuringElement[2, 0] = 0
            structuringElement[2, 2] = 0
    if isFullSized:
        structuringElement = numpy.ones((3, 3, 3), numpy.bool)

    return structuringElement


def dilate_mask(maskData, dilation=1, isFullSized=False, is2D=False):
    structuringElement = get_structuring_element(isFullSized, is2D)
    timesDilated = 0
    while timesDilated < dilation:
        maskData = morphology.binary_dilation(maskData, structure=structuringElement)
        timesDilated += 1
    return maskData


def erode_mask(maskData, erosion=1, isFullSized=False, is2D=False):
    structuringElement = get_structuring_element(isFullSized, is2D)
    timesEroded = 0
    while timesEroded < erosion:
        maskData = morphology.binary_erosion(maskData, structuringElement)
        timesEroded += 1
    return maskData


def open_mask(maskData, times, isFullSized=False, is2D=False):
    maskData = erode_mask(maskData, times, isFullSized, is2D=is2D)
    return dilate_mask(maskData, times, isFullSized, is2D=is2D)


def close_mask(maskData, times, isFullSized=False, is2D=False):
    maskData = dilate_mask(maskData, times, isFullSized=isFullSized, is2D=is2D)
    return erode_mask(maskData, times, isFullSized=isFullSized, is2D=is2D)


def union_masks(masks):
    tempMask = masks[0]
    for i in xrange(1, len(masks)):
        tempMask = numpy.logical_or(tempMask, masks[i])
    return tempMask