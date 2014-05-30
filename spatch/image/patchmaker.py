"""
Created on 17 Dec 2012

@author: Zehan Wang
"""
from __future__ import division
import os
import math

from numpy.lib.stride_tricks import as_strided as ast
from numpy.core.umath import logical_and
import numpy
from spatch.utilities.io import open_image, get_voxel_size
from spatch.utilities.misc import auto_non_background_labels

import mask
from transform import interpolate_to_shape

from spatialcontext import region_dict_from_dt_dict, get_dt_spatial_context_dict, \
    EDT, GDT, GENERIC_SPATIAL_INFO_TYPES, get_generic_spatial_info
from transform import zero_out_boundary, image_boundary_expand
from intensity import rescale_data


__all__ = ["PatchMaker", "DataSetPatchMaker", "get_patches"]


class PatchMaker(object):
    """
    Creates patches from an image
    """

    def __init__(self, imagePath, labelsPath, imageExpand=False, specificLabels=None, boundaryLabels=None,
                 minValue=None, maxValue=None, dtLabelsPath=None, gdtImagePath=None,
                 is2D=False, rescaleIntensities=False):
        """
        Constructor
        if rescale: patches and spatial data will be rescaled to be in range [0, 100]
        """
        self.fileName = os.path.basename(imagePath)
        self.imagePath = imagePath
        self.labelsPath = labelsPath
        self.dtLabelsPath = dtLabelsPath
        # self.image = open_image(imagePath)
        # self.labelsData = numpy.int8(open_image(labelsPath))
        self.imageExpand = imageExpand
        self.minValue = minValue
        self.maxValue = maxValue
        self.specificLabels = specificLabels
        self.boundaryLabels = boundaryLabels

        self.is2D = is2D
        self.rescaleIntensities = rescaleIntensities
        self.gdtImagePath = gdtImagePath

    def _get_labels_data(self):
        labelsData = open_image(self.labelsPath)
        if self.imageExpand:
            labelsData = image_boundary_expand(labelsData, useGradient=False, is2D=self.is2D)
        return labelsData

    def _get_image_data(self):
        imageData = open_image(self.imagePath)
        if self.imageExpand:
            imageData = image_boundary_expand(imageData, is2D=self.is2D)
        return imageData

    def _get_dt_image_data(self, spatialInfoType, imageData):
        dtImage = None
        if spatialInfoType == GDT:
            if self.gdtImagePath is not None:
                dtImage = open_image(self.gdtImagePath)
            elif self.imageExpand:
                dtImage = imageData[1:-1, 1:-1, 1:-1]
            else:
                dtImage = imageData
        return dtImage

    def _get_dt_voxel_size(self):
        if self.gdtImagePath is not None:
            return get_voxel_size(self.gdtImagePath)
        else:
            return get_voxel_size(self.labelsPath)

    def get_patch_dict(self, patchSize, spatialWeight=0, boundaryDilation=None, spatialRegionLabels=None,
                       spatialRegionIndex=None, spatialLabels=None, dtSeeds=None,
                       labelErosion=0, boundaryClipSize=0, spatialInfoType=EDT, roiMask=None, separateSpatial=False,
                       includePatchSizeKey=False):
        """
        @param patchSize: int or tuple to determine size of patch. if int, assume isotropic
        @param spatialWeight: spatial weighting to apply if using spatial info
        @param boundaryDilation: boundary around labels (if getting boundary refinement patches)
        @param spatialRegionIndex: (index of region -> indexed by edt labels) or counts from 0 if spatial labels not
        provided
        @param spatialLabels: labels to use to get edt/gdt based spatial info, if none
        @param labelErosion:
        @param boundaryClipSize:
        @param roiMask:
        @return: a dictionary of {label: patches}
        """
        imageData = self._get_image_data()
        labelsData = open_image(self.labelsPath)

        # put check in that transformed atlas actually overlaps
        if labelsData.max() == 0:
            return None
        elif spatialLabels is not None:
            uniqueLabels = numpy.unique(labelsData)
            if set(uniqueLabels).intersection(set(spatialLabels)) == set():
                return None

        voxelSize = self._get_dt_voxel_size()
        getBoundaryPatches = roiMask is None

        if self.imageExpand:
            try:
                boundaryClipSize -= 1
            except TypeError:
                boundaryClipSize = numpy.asarray(boundaryClipSize) - 1

        spatialLabelDict = None
        spatialData = None

        # get spatial information
        if spatialWeight > 0 and spatialInfoType is not None:
            if spatialInfoType in GENERIC_SPATIAL_INFO_TYPES:
                spatialData = get_generic_spatial_info(imageData, spatialInfoType)
            else:
                if dtSeeds is not None:
                    dtLabelsData = dtSeeds
                else:
                    dtLabelsData = labelsData
                    if self.dtLabelsPath is not None:
                        dtLabelsData = open_image(self.dtLabelsPath)
                        if dtLabelsData.shape != labelsData.shape:
                            dtLabelsData = interpolate_to_shape(dtLabelsData, labelsData.shape)

                dtImage = self._get_dt_image_data(spatialInfoType, imageData)

                spatialLabelDict = get_dt_spatial_context_dict(dtLabelsData, spatialInfoType,
                                                               spatialLabels=spatialLabels,
                                                               voxelSize=voxelSize, labelErosion=labelErosion,
                                                               boundaryClipSize=boundaryClipSize,
                                                               imageData=dtImage, is2D=self.is2D,
                                                               imageExpand=self.imageExpand)

                spatialData = spatialLabelDict.values()
            spatialData = numpy.asarray(spatialData) * spatialWeight

        # get regional mask
        if roiMask is not None:
            roiMask = interpolate_to_shape(roiMask, imageData.shape)

        if spatialRegionIndex is not None:
            if spatialRegionLabels is None:
                spatialRegionLabels = spatialLabels
            if spatialLabelDict is None:
                dtImage = self._get_dt_image_data(spatialInfoType, imageData)
                dtLabelsData = labelsData
                if self.dtLabelsPath is not None:
                    dtLabelsData = open_image(self.dtLabelsPath)
                spatialLabelDict = get_dt_spatial_context_dict(dtLabelsData, spatialInfoType,
                                                               spatialLabels=spatialLabels,
                                                               voxelSize=voxelSize, labelErosion=labelErosion,
                                                               boundaryClipSize=boundaryClipSize,
                                                               imageData=dtImage, is2D=self.is2D,
                                                               imageExpand=self.imageExpand)

            regionMask = region_dict_from_dt_dict(dict((l, spatialLabelDict[l]) for l in spatialRegionLabels),
                                                  regionalOverlap=boundaryDilation,
                                                  specificRegionIndex=spatialRegionIndex, is2D=self.is2D)
            if roiMask is None:
                roiMask = regionMask
            else:
                roiMask = logical_and(roiMask, regionMask)

        # get overall label and data masks
        if self.specificLabels is None:
            self.specificLabels = numpy.unique(labelsData)
        if self.boundaryLabels is None:
            self.boundaryLabels = auto_non_background_labels(labelsData)

        if self.imageExpand:
            labelsData = image_boundary_expand(labelsData, useGradient=False, is2D=self.is2D)

        labelMasks = mask.get_label_masks(labelsData, self.specificLabels)
        boundarySize = boundaryDilation
        if not getBoundaryPatches:
            boundarySize = None

        maskData = mask.get_data_mask(imageData, labelsData, boundarySize=boundarySize,
                                      specificLabels=self.boundaryLabels, roiMask=roiMask,
                                      minValue=self.minValue, maxValue=self.maxValue)

        if maskData is not None:
            labelMasks = [logical_and(m, maskData) for m in labelMasks]
            if self.rescaleIntensities:
                imageData = rescale_data(imageData, maskData=maskData)

        patchDict = dict((self.specificLabels[i], get_patches(imageData, patchSize, labelMasks[i],
                                                              spatialData=spatialData, separateSpatial=separateSpatial))
                         for i in xrange(len(labelMasks)))
        if includePatchSizeKey:
            patchDict["patchSize"] = patchSize
        return patchDict


def patch_view(data, patchSize=(3, 3, 3)):
    """Returns a view of overlapping patches from the data"""

    if isinstance(patchSize, int):
        patchSize = (patchSize,) * 3
    elif len(patchSize) == 1:
        patchSize = (patchSize[0],) * 3
    else:
        patchSize = tuple(patchSize)

    shape = tuple(numpy.asarray(data.shape) - numpy.asarray(patchSize) + 1) + patchSize

    strides = data.strides + data.strides
    patchMatrix = ast(data, shape=shape, strides=strides)
    return patchMatrix


def non_overlapping_patch_view(data, patchSize):
    if isinstance(patchSize, int):
        patchSize = (patchSize,) * 3
    elif len(patchSize) == 1:
        patchSize = (patchSize[0],) * 3

    shape = tuple(numpy.int(numpy.asarray(data.shape) / numpy.asarray(patchSize))) + patchSize

    strides = tuple(numpy.asarray(data.strides) * numpy.asarray(patchSize)) + data.strides
    patchMatrix = ast(data, shape=shape, strides=strides)
    return patchMatrix


def get_patches(imageData, patchSize, maskData=None, spatialData=None, separateSpatial=False,
                verbose=False, overlapping=True):
    if overlapping:
        return get_patches_from_ast_data(patch_view(imageData, patchSize), maskData=maskData,
                                         spatialData=spatialData, verbose=verbose, separateSpatial=separateSpatial)
    else:
        #TODO may need to trim maskData and spatial data if shape not exact multiple of patchSize
        return get_patches_from_ast_data(non_overlapping_patch_view(imageData, patchSize), maskData=maskData,
                                         spatialData=spatialData, verbose=verbose, separateSpatial=separateSpatial)


def get_non_overlapping_patches(imageData, patchSize, maskData=None, spatialData=None, verbose=False):
    return get_patches_from_ast_data(non_overlapping_patch_view(imageData, patchSize), maskData=maskData,
                                     spatialData=spatialData, verbose=verbose)


def get_patches_from_ast_data(imageData, maskData=None, spatialData=None, verbose=False, separateSpatial=False):
    """"
        Fast efficient patch extraction
        Assumes imageData is patch_view on data
        maskData is of the same shape as original image data (not as patch view)
        Assumes spatialData is a list of 3D matrices if provided
        If normaliseFeatures: imageData and spatialData will each be normalised to be in range [0, 100]
            by max of imageData and spatialData respectively

    """
    # align mask to strided data
    offsetX = int(math.floor(imageData.shape[3] / 2))
    offsetY = int(math.floor(imageData.shape[4] / 2))
    offsetZ = int(math.floor(imageData.shape[5] / 2))
    if maskData is None:
        maskData = numpy.ones(numpy.asarray(imageData.shape[:3]) + numpy.asarray(imageData.shape[3:]) - 1, numpy.bool)

    if offsetZ > 0:
        alignedMask = maskData[offsetX:-offsetX, offsetY:-offsetY, offsetZ:-offsetZ]
    else:
        alignedMask = maskData[offsetX:-offsetX, offsetY:-offsetY]
    numPatches = numpy.count_nonzero(alignedMask)
    patches = imageData[alignedMask].reshape(numPatches, numpy.prod(imageData.shape[3:]))

    if spatialData is not None:
        # don't include anything outside boundaries
        maskData = zero_out_boundary(maskData, (offsetX, offsetY, offsetZ))
        spatialData = numpy.asarray(spatialData)
        try:
            spatialInfo = spatialData[..., maskData]
        except:
            print "Shapes:", spatialData.shape, maskData.shape
            raise
        spatialInfo = numpy.atleast_2d(numpy.transpose(spatialInfo))
        # assumes returned spatialInfo will be in the same ordering as patches
        if separateSpatial:
            patches = (patches, spatialInfo)
        else:
            patches = numpy.append(patches, spatialInfo, 1)
    if verbose:
        print "Number of patches created:", numPatches
    return patches