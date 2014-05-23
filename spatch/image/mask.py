"""
Created on 22 Mar 2012

@author: zw606
"""
import glob
import os

from skimage.filter import threshold_otsu
from scipy.ndimage import morphology, gaussian_filter, binary_fill_holes
import numpy

from utilities.misc import auto_non_background_labels
from utilities.io import open_image, get_affine, save_3d_data


def get_masked_data_from_file(imagePath, labelsPath, boundarySize=None, specificLabels=None, roiMask=None,
                              minValue=None, maxValue=None):
    imData = open_image(imagePath)
    if boundarySize:
        labelsData = open_image(labelsPath)
    else:
        labelsData = None
    return get_masked_data(imData, labelsData, boundarySize=boundarySize, specificLabels=specificLabels,
                           roiMask=roiMask, minValue=minValue, maxValue=maxValue)


def get_masked_data(imData, labels, boundarySize=None, specificLabels=None, roiMask=None,
                    minValue=None, maxValue=None):
    if minValue is not None:
        imData *= imData > minValue
    if maxValue is not None:
        imData *= imData < maxValue
    if roiMask is not None:
        imData *= roiMask
    if boundarySize:
        imData *= get_boundary_mask(labels, boundarySize, specificLabels=specificLabels)
    return imData


def get_data_mask_from_file(imagePath, labelsPath, boundarySize=None, specificLabels=None, roiMask=None,
                            minValue=None, maxValue=None):
    imData = open_image(imagePath)
    if boundarySize:
        labelsData = open_image(labelsPath)
    else:
        labelsData = None
    return get_data_mask(imData, labelsData, boundarySize=boundarySize, specificLabels=specificLabels,
                         roiMask=roiMask, minValue=minValue, maxValue=maxValue)


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


def non_zero_locations(maskData):
    locationList = numpy.atleast_2d(numpy.squeeze(numpy.transpose(numpy.nonzero(maskData))))
    return list(locationList)


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


def get_surface_mask(maskData):
    return maskData - erode_mask(maskData)


def get_surface_mask2(maskData):
    return dilate_mask(maskData) - maskData


def get_label_masks(labelledData, specificLabels=None, cubic=False):
    """ returns list of boundary masks for specified labels"""
    if specificLabels is None:
        specificLabels = numpy.unique(labelledData)

    return [labelledData == label for label in specificLabels]


def pad_otsu_background(data, gaussianFilterSigma=2, maskClosing=4, padding=-1):
    # for some reason, have to convert to uint for otsu to work
    data2 = numpy.uint32(numpy.round(data * 10))

    threshold = numpy.percentile(data2, 90)
    print "thresholding values at 90th percentile:", threshold
    thresholdMask = data2 > threshold
    data2[thresholdMask] = threshold
    if gaussianFilterSigma is not None:
        data2 = gaussian_filter(data2, gaussianFilterSigma)
    noPadMask = data2 >= threshold_otsu(data2)
    if maskClosing is not None:
        noPadMask = close_mask(noPadMask, maskClosing)
    noPadMask = binary_fill_holes(noPadMask)
    paddingMask = noPadMask == 0
    data[paddingMask] = padding
    return data


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


def create_masks_from_dir(labelsDir, savePath, dilation, closing, specificLabels, fileContains="*.nii.gz",
                          intersectionCheck=False,
                          padding=None, dataPath=None):
    labels = glob.glob(labelsDir + fileContains)
    data = None
    if padding is not None:
        data = glob.glob(dataPath + fileContains)
    print "There are", len(labels), "labels"
    return create_union_masks(labels, savePath, dilation, closing, specificLabels, intersectionCheck=intersectionCheck,
                              padding=padding, dataFiles=data)


def create_intersections_masks(labelsPath, savePath, erosion=1, opening=0, specificLabels=None, is2D=False,
                               fileContains="*.nii.gz"):
    masks = []
    labels = glob.glob(labelsPath + fileContains)
    tempMask = open_image(labels[0]).copy()
    affine = get_affine(labels[0])
    numLabels = numpy.max(tempMask)

    if specificLabels is not None:
        for x in specificLabels:
            if x > numLabels:
                specificLabels.remove(x)
        labelsToMask = specificLabels
    else:
        labelsToMask = range(1, numLabels + 1)

    print "Labels to mask:", labelsToMask
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for i in xrange(len(labelsToMask)):
        # create mask for each label
        masks.append(tempMask == labelsToMask[i])
        print "Initial mask size:", numpy.sum(masks[i])
    for i in xrange(1, len(labels)):
        print "Adding mask", i
        tempMask = open_image(labels[i])
        notReported = True
        for j in xrange(len(labelsToMask)):
            masks[j] = numpy.logical_and(masks[j], tempMask == labelsToMask[j])
            if numpy.sum(masks[j]) == 0 and notReported:
                print "no further intersection at mask", i
                print labels[i]
                notReported = False

    maskSizes = []
    for i in xrange(len(labelsToMask)):
        masks[i] = erode_mask(masks[i], erosion, is2D=is2D)
        masks[i] = open_mask(masks[i], opening, is2D=is2D)
        maskSize = numpy.sum(masks[i])
        print "Mask size:", maskSize
        maskSizes.append(maskSize)
        fileName = str(labelsToMask[i]) + ".nii.gz"
        save_3d_data(numpy.int16(masks[i]), affine, savePath + fileName)


def create_union_masks(labels, savePath, dilation=1, closing=0, specificLabels=None, intersectionCheck=True,
                       numLabelsCheck=4,
                       padding=None, dataFiles=None, is2D=False):
    """Creates a mask which is the union of all data"""
    masks = []
    tempMask = open_image(labels[0])
    affine = get_affine(labels[0])
    numLabels = numpy.max(tempMask)
    if padding is not None:
        paddedAreas = open_image(dataFiles[0]) == padding

    if numLabelsCheck:
        if numLabels < numLabelsCheck:
            print "Mask has no intersection with other masks! Probable registration failure!"
            return 0

    if specificLabels is not None:
        for x in specificLabels:
            if x > numLabels:
                specificLabels.remove(x)
        labelsToMask = specificLabels
    else:
        labelsToMask = range(1, numLabels + 1)

    if not os.path.exists(savePath + "Unions/"):
        os.makedirs(savePath + "Unions/")

    for i in xrange(len(labelsToMask)):
        # create mask for each label
        masks.append(tempMask == labelsToMask[i])
        print "Initial mask size:", numpy.sum(masks[i])

    for i in xrange(1, len(labels)):
        print "Adding mask", i
        tempMask = open_image(labels[i])
        if numLabelsCheck:
            if numLabels < numLabelsCheck:
                print "Mask has no labels! Probable registration failure!"
                return 0
        for j in xrange(len(labelsToMask)):
            masks[j] = numpy.logical_or(masks[j], tempMask == labelsToMask[j])
        if padding is not None:
            paddedAreas = numpy.logical_or(paddedAreas, open_image(dataFiles[i]) == padding)
    if intersectionCheck:
        for i in xrange(len(labels)):
            tempMask = open_image(labels[i])
            for j in xrange(len(labelsToMask)):
                if not has_intersection(masks[j], tempMask == labelsToMask[j]):
                    print "Mask has no intersection with other masks! Probable registration failure!"
                    print "failed image:", labels[i]
                    return 0
    maskSizes = []
    for i in xrange(len(labelsToMask)):
        closedMask = close_mask(masks[i], closing)
        masks[i] = numpy.logical_or(masks[i], closedMask)
        masks[i] = dilate_mask(masks[i], dilation, is2D=is2D)

        if padding is not None:
            masks[i] -= masks[i] * paddedAreas

        maskSize = numpy.sum(masks[i])
        print "Mask size:", maskSize
        maskSizes.append(maskSize)
        fileName = str(labelsToMask[i]) + ".nii.gz"
        save_3d_data(numpy.int16(masks[i]), affine, savePath + "Unions/" + fileName)
    return maskSizes


def has_intersection(unionMask, queryMask):
    """returns true if the intersection is not 0"""

    intersection = numpy.logical_and(unionMask, queryMask)
    if numpy.sum(intersection) > numpy.sum(queryMask):
        return True
    return False


def intersection_masks(masks):
    tempMask = masks[0]
    for i in xrange(1, len(masks)):
        tempMask = numpy.logical_and(tempMask, masks[i])
    return tempMask


def union_masks(masks):
    tempMask = masks[0]
    for i in xrange(1, len(masks)):
        tempMask = numpy.logical_or(tempMask, masks[i])
    return tempMask


def remove_zeros(data, mask):
    data = data * mask
    data = data.ravel()
    blanks = numpy.flatnonzero(data == 0)
    return numpy.delete(data, blanks)


def remove_zeros3(data, mask):
    data += 1
    data = data * mask
    data -= (mask == 0)
    data = data.ravel()
    blanks = numpy.flatnonzero(data == -1)
    data = numpy.delete(data, blanks)
    data -= 1
    return data


def remove_values(data, value):
    data = data.ravel()
    toBeRemoved = numpy.flatnonzero(data == value)
    return numpy.delete(data, toBeRemoved)


def union_masks_from_files(maskFiles, dilation, closing, savePath, padding=None, dataPath=None,
                           cubicDilate=False, is2D=False, dataFileName="*nii.gz"):
    tempMask = open_image(maskFiles[0]).copy()
    affine = get_affine(maskFiles[0])
    if padding is not None:
        dataFiles = glob.glob(dataPath + dataFileName)
        paddedAreas = open_image(dataFiles[0]) == padding
        for i in xrange(1, len(dataFiles)):
            paddedAreas = numpy.logical_or(paddedAreas, open_image(dataFiles[i]) == padding)
    for i in xrange(1, len(maskFiles)):
        tempMask = numpy.logical_or(tempMask, open_image(maskFiles[i]))

    if dilation > 0:
        tempMask = dilate_mask(tempMask, dilation, cubicDilate, is2D=is2D)
    if closing > 0:
        tempMask2 = close_mask(tempMask, closing, is2D=is2D)
        tempMask = numpy.logical_or(tempMask, tempMask2)
    if padding is not None:
        tempMask -= tempMask * paddedAreas
    save_3d_data(numpy.int16(tempMask), affine, savePath)
    print "Unioned mask size:", numpy.sum(tempMask)


def split_mask_from_file(maskPath, numIterations, savePath, intersectionMaskPath=None):
    maskData = open_image(maskPath)
    if intersectionMaskPath is not None:
        maskData -= open_image(intersectionMaskPath)
    affine = get_affine(maskPath)
    masksToSplit = [maskData]
    completedMasks = []
    for i in xrange(numIterations):
        print "Splitting masks, iteration", i + 1
        while len(masksToSplit) > 0:
            maskData = masksToSplit.pop()
            mask1, mask2 = split_mask(maskData)
            completedMasks.append(mask1)
            completedMasks.append(mask2)
        masksToSplit = completedMasks
        completedMasks = []
    for i in xrange(len(masksToSplit)):
        save_3d_data(masksToSplit[i], affine, savePath + str(i) + ".nii.gz")
        print "Mask size:", numpy.sum(masksToSplit[i])


def split_mask(maskData):
    """
        Splits the mask into two
    """
    # find boundingBox
    mins, maxes = get_bounding_box(maskData)
    mins = numpy.asarray(mins)
    maxes = numpy.asarray(maxes)
    diffs = maxes - mins
    splitIndex = numpy.argmax(diffs)
    centroid = find_centroid(maskData)
    splitNum = centroid[splitIndex]

    splitMins = mins.copy()
    splitMins[splitIndex] = splitNum
    splitMaxes = maxes.copy()
    splitMaxes[splitIndex] = splitNum

    mask1 = maskData.copy()
    mask1[splitMins[0]:maxes[0], splitMins[1]:maxes[1], splitMins[2]:maxes[2]] = 0
    mask2 = maskData.copy()
    mask2[mins[0]:splitMaxes[0], mins[1]:splitMaxes[1], mins[2]:splitMaxes[2]] = 0

    return mask1, mask2


def find_centroid(maskData):
    coordinates = [[], [], []]
    for i in xrange(maskData.shape[0]):
        for j in xrange(maskData.shape[1]):
            for k in xrange(maskData.shape[2]):
                if maskData[i, j, k] == 1:
                    coordinates[0].append(i)
                    coordinates[1].append(j)
                    coordinates[2].append(k)

    centroid = (int(round(numpy.average(coordinates[0]))),
                int(round(numpy.average(coordinates[1]))),
                int(round(numpy.average(coordinates[2]))))
    return centroid


def confined_dilation_from_files(maskPath, dilation, confinementMaskPath, savePath, fileName="*.nii.gz",
                                 intersectionMaskPath=None):
    masks = glob.glob(maskPath + fileName)
    confineMentMask = open_image(confinementMaskPath)
    if intersectionMaskPath is not None:
        confineMentMask -= open_image(intersectionMaskPath)
    for x in masks:
        baseName = os.path.basename(x)
        maskData = confined_dilation(open_image(x), dilation, confineMentMask)
        save_3d_data(maskData, get_affine(x), savePath + baseName)
        print "Dilated mask size:", numpy.sum(maskData)


def confined_dilation(maskData, dilation, confinementMask, is2D=False):
    maskData = dilate_mask(maskData, dilation, is2D=is2D)
    maskData *= confinementMask
    return numpy.int16(maskData)


def remove_padded_areas_from_files(dataPath, masksPath, savePath, padding=-1, dataFileName="*.nii.gz",
                                   maskFileName="*.nii.gz"):
    print "Calculating padded areas..."
    dataFiles = glob.glob(dataPath + dataFileName)
    paddedAreas = open_image(dataFiles[0]) == padding
    for i in xrange(1, len(dataFiles)):
        paddedAreas = numpy.logical_or(paddedAreas, open_image(dataFiles[i]) == padding)

    print "Removing padded areas..."
    maskFiles = glob.glob(masksPath + maskFileName)
    for maskFile in maskFiles:
        maskData = open_image(maskFile)
        maskData -= maskData * paddedAreas
        save_3d_data(maskData, get_affine(maskFile), savePath + os.path.basename(maskFile))
        print "New mask size:", numpy.sum(maskData)