"""
Created on 17 Dec 2012

@author: zw606
"""
from __future__ import division
import os
import cPickle as pickle
import math

from numpy.lib.stride_tricks import as_strided as ast
from numpy.core.umath import logical_and
import numpy
from utilities.io import auto_make_dir, open_image, get_voxel_size, construct_datafiles_list
from utilities.misc import auto_non_background_labels

import mask
from transform import interpolate_to_shape

from spatialcontext import multi_label_edt_dict, region_dict_from_dt_dict, get_dt_spatial_context_dict, \
    EDT, GDT, dist_to_y_2d_centre_spatial_info, dist_to_centre_spatial_info, get_coordinates, get_generic_spatial_info
from spatch.image.resize import zero_out_boundary, image_boundary_expand
from intensity import rescale_data


COORDINATES = "coordinates"
NORMALISED_COORDINATES = "normalised-coordinates"
COORDINATES_2D = "coordinates-2d"
NORMALISED_COORDINATES_2D = "normalised-coordinates-2d"
DIST_CENTRE = "dist-to-centre"
NORMALISED_DIST_CENTRE = "normalised-dist-to-centre"
NORMALISED_DIST_CENTRE_MASS = "normalised-dist-to-centre-mass"
NORMALISED_COORDINATES_CENTRE_MASS = "normalised-coordinates-to-centre-mass"
SPATIAL_INFO_TYPES = [EDT, GDT, COORDINATES, NORMALISED_COORDINATES, DIST_CENTRE, NORMALISED_DIST_CENTRE,
                      NORMALISED_DIST_CENTRE_MASS, NORMALISED_COORDINATES_CENTRE_MASS]
GENERIC_SPATIAL_INFO_TYPES = [COORDINATES, NORMALISED_COORDINATES, DIST_CENTRE, NORMALISED_DIST_CENTRE,
                              NORMALISED_DIST_CENTRE_MASS, NORMALISED_COORDINATES_CENTRE_MASS]


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

    def get_labels_data(self):
        labelsData = open_image(self.labelsPath)
        if self.imageExpand:
            labelsData = image_boundary_expand(labelsData, useGradient=False, is2D=self.is2D)
        return labelsData

    def get_image_data(self):
        imageData = open_image(self.imagePath)
        if self.imageExpand:
            imageData = image_boundary_expand(imageData, is2D=self.is2D)
        return imageData

    def __get_dt_image_data(self, spatialInfoType, imageData):
        dtImage = None
        if spatialInfoType == GDT:
            if self.gdtImagePath is not None:
                dtImage = open_image(self.gdtImagePath)
            elif self.imageExpand:
                dtImage = imageData[1:-1, 1:-1, 1:-1]
            else:
                dtImage = imageData
        return dtImage

    def get_dt_voxel_size(self):
        if self.gdtImagePath is not None:
            return get_voxel_size(self.gdtImagePath)
        else:
            return get_voxel_size(self.labelsPath)

    def get_patch_dict2(self, patchSize, spatialWeight=0, boundaryDilation=None, spatialRegionLabels=None,
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
        imageData = self.get_image_data()
        labelsData = open_image(self.labelsPath)

        # put check in that transformed atlas actually overlaps
        if labelsData.max() == 0:
            return None
        elif spatialLabels is not None:
            uniqueLabels = numpy.unique(labelsData)
            if set(uniqueLabels).intersection(set(spatialLabels)) == set():
                return None

        voxelSize = self.get_dt_voxel_size()
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

                dtImage = self.__get_dt_image_data(spatialInfoType, imageData)

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
                dtImage = self.__get_dt_image_data(spatialInfoType, imageData)
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

    def get_patch_dict(self, patchSize, maskData=None, minValue=None, maxValue=None):
        """
            returns a dictionary {label : [patches]}

        @param patchSize: int or tuple to determine size of patch. if int, assume isotropic
        @param maskData: mask to use (numpy array)
        @param minValue: minimum value in data to include
        @param maxValue: max value in data to include
        """
        imageData = self.get_image_data()
        labelsData = self.get_labels_data()
        maskData = mask.get_min_max_mask(imageData, minValue, maxValue, maskData)
        labelMasks = mask.get_label_masks(labelsData, self.specificLabels)
        if maskData is not None:
            labelMasks = [logical_and(m, maskData) for m in labelMasks]
        patchDict = dict((self.specificLabels[i], get_patches(imageData, patchSize, labelMasks[i], verbose=True))
                         for i in xrange(len(labelMasks)))
        patchDict["patchSize"] = patchSize
        return patchDict

    def get_coordinate_patch_dict(self, patchSize, maskData=None, minValue=None, maxValue=None, normaliseSpatial=False):
        labelsData = self.get_labels_data()
        spatialInfo = get_coordinates(labelsData, normalise=normaliseSpatial)
        return self.get_spatial_patch_dict(patchSize, spatialInfo, maskData, minValue, maxValue)

    def get_centre_dist_patch_dict(self, patchSize,
                                   maskData=None, minValue=None, maxValue=None, normaliseSpatial=False):
        labelsData = self.get_labels_data()
        spatialInfo = dist_to_centre_spatial_info(labelsData, normalise=normaliseSpatial)
        return self.get_spatial_patch_dict(patchSize, spatialInfo, maskData, minValue, maxValue)

    def get_y_centre_dist_patch_dict(self, patchSize,
                                     maskData=None, minValue=None, maxValue=None, normaliseSpatial=False):
        labelsData = self.get_labels_data()
        spatialInfo = dist_to_y_2d_centre_spatial_info(labelsData, normalise=normaliseSpatial)
        return self.get_spatial_patch_dict(patchSize, spatialInfo, maskData, minValue, maxValue)

    def get_boundary_patch_dict(self, patchSize, boundaryDilation,
                                maskData=None, minValue=None, maxValue=None):
        """
            returns a dictionary {label : [patches]} but where patches come from the boundary of the label
            the boundary thickness = 2* boundaryDilation
        """
        labelsData = self.get_labels_data()
        if maskData is None:
            maskData = mask.get_boundary_mask(labelsData, boundaryDilation, self.boundaryLabels)
        else:
            maskData *= mask.get_boundary_mask(labelsData, boundaryDilation, self.boundaryLabels)

        return self.get_patch_dict(patchSize, maskData, minValue, maxValue)

    def get_spatial_patch_dict(self, patchSize, spatialInfo,
                               maskData=None, minValue=None, maxValue=None):
        """
            returns a dictionary {label : [Spatially aware patches]}
            spatialInfo must be of compatible shape to image (need to expand if imageExpand)
                - spatial info is appended to the patch intensities
        """
        imageData = self.get_image_data()
        labelsData = self.get_labels_data()
        maskData = mask.get_min_max_mask(imageData, minValue, maxValue, maskData)
        labelMasks = mask.get_label_masks(labelsData, self.specificLabels)
        if maskData is not None:
            labelMasks = [logical_and(m, maskData) for m in labelMasks]
        patchDict = dict((self.specificLabels[i], get_patches(imageData, patchSize, labelMasks[i],
                                                              spatialData=spatialInfo, verbose=True))
                         for i in xrange(len(labelMasks)))
        patchDict["patchSize"] = patchSize
        return patchDict

    def get_boundary_spatial_patch_dict(self, patchSize, spatialInfo, boundaryDilation,
                                        maskData=None, minValue=None, maxValue=None):
        """
            returns a dictionary {label : [patches]} but where patches come from the boundary of the label
            the boundary thickness = 2* boundaryDilation
        """
        labelsData = self.get_labels_data()
        if maskData is None:
            maskData = mask.get_boundary_mask(labelsData, boundaryDilation, self.boundaryLabels)
        else:
            maskData = logical_and(mask.get_boundary_mask(labelsData, boundaryDilation, self.boundaryLabels),
                                   maskData)

        return self.get_spatial_patch_dict(patchSize, spatialInfo,
                                           maskData=maskData, minValue=minValue, maxValue=maxValue)


class DataSetPatchMaker(object):
    def __init__(self, imagesPath, labelsPath, imageExpand, nameContains="*.nii.gz", specificLabels=None,
                 boundaryLabels=None, preEdtErosion=0):
        imageLabelPairs = construct_datafiles_list(imagesPath, labelsPath, nameContains)
        self.dataset = [PatchMaker(imagePath, labelPath, imageExpand,
                                   specificLabels=specificLabels, boundaryLabels=boundaryLabels)
                        for imagePath, labelPath in imageLabelPairs]
        self.imageExpand = imageExpand
        self.preEdtErosion = preEdtErosion

    def save_patch_dicts(self, patchSize, savePath, minValue=None, maxValue=None):
        for x in self.dataset:
            pickle.dump(x.get_patch_dict(patchSize, minValue=minValue, maxValue=maxValue),
                        open(savePath + x.fileName + ".dict", "w"), protocol=2)

    def save_coordinate_patch_dicts(self, patchSize, savePath, minValue=None, maxValue=None, normaliseSpatial=False):
        for x in self.dataset:
            pickle.dump(x.get_coordinate_patch_dict(patchSize, minValue=minValue, maxValue=maxValue,
                                                    normaliseSpatial=normaliseSpatial),
                        open(savePath + x.fileName + ".dict", "w"), protocol=2)

    def save_centre_dist_patch_dicts(self, patchSize, savePath, minValue=None, maxValue=None,
                                     normaliseSpatial=False):
        for x in self.dataset:
            patchDict = x.get_centre_dist_patch_dict(patchSize, minValue=minValue, maxValue=maxValue,
                                                     normaliseSpatial=normaliseSpatial)
            pickle.dump(patchDict, open(savePath + x.fileName + ".dict", "w"), protocol=2)

    def save_y_centre_dist_patch_dicts(self, patchSize, savePath, minValue=None, maxValue=None,
                                       normaliseSpatial=False):
        for x in self.dataset:
            patchDict = x.get_y_centre_dist_patch_dict(patchSize, minValue=minValue, maxValue=maxValue,
                                                       normaliseSpatial=normaliseSpatial)
            pickle.dump(patchDict, open(savePath + x.fileName + ".dict", "w"), protocol=2)

    def save_boundary_patch_dicts(self, patchSize, savePath, boundaryDilation,
                                  minValue=None, maxValue=None):
        for x in self.dataset:
            pickle.dump(x.get_boundary_patch_dict(patchSize, boundaryDilation, minValue=minValue, maxValue=maxValue),
                        open(savePath + x.fileName + ".dict", "w"), protocol=2)

    def save_edt_regional_patch_dicts(self, patchSize, savePath, edtLabelsPath,
                                      edtLabels=None, minValue=None, maxValue=None):
        for x in self.dataset:
            labelsData = open_image(edtLabelsPath + x.fileName)
            voxelSpacing = get_voxel_size(edtLabelsPath + x.fileName)
            edtDistances = multi_label_edt_dict(labelsData, specificLabels=edtLabels, voxelSpacing=voxelSpacing)
            regionMasks = region_dict_from_dt_dict(edtDistances).values()

            for i in range(len(regionMasks)):
                auto_make_dir(savePath + str(i) + "/")
                pickle.dump(x.get_patch_dict(patchSize, maskData=regionMasks[i], minValue=minValue, maxValue=maxValue),
                            open(savePath + str(i) + "/" + x.fileName + ".dict", "w"), protocol=2)

    def save_edt_regional_boundary_patch_dicts(self, patchSize, savePath, boundaryDilation, edtLabelsPath,
                                               edtLabels=None, minValue=None, maxValue=None):
        for x in self.dataset:
            labelsData = open_image(edtLabelsPath + x.fileName)
            voxelSpacing = get_voxel_size(edtLabelsPath + x.fileName)
            edtDistances = multi_label_edt_dict(labelsData, specificLabels=edtLabels, voxelSpacing=voxelSpacing)
            regionMasks = region_dict_from_dt_dict(edtDistances).values()

            for i in range(len(regionMasks)):
                auto_make_dir(savePath + str(i) + "/")
                pickle.dump(x.get_boundary_patch_dict(patchSize, boundaryDilation,
                                                      maskData=regionMasks[i], minValue=minValue, maxValue=maxValue),
                            open(savePath + str(i) + "/" + x.fileName + ".dict", "w"), protocol=2)

    def save_edt_regional_boundary_spatial_patch_dicts(self, patchSize, savePath, boundaryDilation,
                                                       edtLabelsPath,
                                                       edtLabels=None, spatialWeight=1, regionalOverlap=0,
                                                       clipLabelEdt=False, minValue=None, maxValue=None,
                                                       clipPatchSize=None):
        labelClipping = 0
        if clipLabelEdt:
            if clipPatchSize is None:
                labelClipping = int(math.floor(patchSize / 2))
            else:
                labelClipping = int(math.floor(clipPatchSize / 2))
            print "Label Clipping enabled - clipping boundary:", labelClipping

        for x in self.dataset:
            labelsData = open_image(edtLabelsPath + x.fileName)
            voxelSpacing = get_voxel_size(edtLabelsPath + x.fileName)
            edtDistances = multi_label_edt_dict(labelsData, specificLabels=edtLabels, voxelSpacing=voxelSpacing,
                                                imageBoundaryClipping=labelClipping)
            regionMasks = region_dict_from_dt_dict(edtDistances).values()
            edtResults = edtDistances.values()

            if spatialWeight != 1:
                edtResults *= spatialWeight
            for i, regionMask in enumerate(regionMasks):
                auto_make_dir(savePath + str(i) + "/")
                pickle.dump(x.get_boundary_spatial_patch_dict(patchSize, edtResults, boundaryDilation,
                                                              maskData=regionMask, minValue=minValue,
                                                              maxValue=maxValue),
                            open(savePath + str(i) + "/" + x.fileName + ".dict", "w"), protocol=2)


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


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sw", dest="spatialWeight", type=float, default=0,
                        help="Set Location Weighting")
    parser.add_argument("--dilation", dest="dilation", type=int, default=2,
                        help="Set Mask dilation for area to build tree in")
    parser.add_argument("--preEdtErosion", dest="preEdtErosion", type=int, default=0,
                        help="Erosion to perform on labels prior to EDT operations")
    parser.add_argument("--regionalOverlap", dest="regionalOverlap", type=int, default=0,
                        help="Amount of overlap between regions")

    parser.add_argument("--patchSize", dest="patchSize", type=int, default=7, nargs="+",
                        help="Set the patch size to use")
    parser.add_argument("--np", dest="numProcessors", type=int, default=4,
                        help="Set the number of processors to use")

    parser.add_argument("--dp", dest="dataPath",
                        help="Set the data path")
    parser.add_argument("--lp", dest="labelsPath",
                        help="Set the labels path")
    parser.add_argument("--edtLP", dest="edtLabelsPath",
                        help="Set the labels path")
    parser.add_argument("--sp", dest="savePath",
                        help="Set the save path")
    parser.add_argument("--tempFolder", dest="tempFolder",
                        help="Set temp folder path")
    parser.add_argument("--imageExpand", dest="imageExpand", default=False, action="store_true",
                        help="Expands image boundary by 1 in each direction on the image "
                             "(if image very small may be useful)")
    parser.add_argument("--ski10Config", dest="ski10Config", default=False, action="store_true",
                        help="Set up for ski10 config with bone labels for EDT")
    parser.add_argument("--ski10Config2", dest="ski10Config2", default=False, action="store_true",
                        help="Set up for ski10 config with all labels for EDT")
    parser.add_argument("--abdominalConfig", dest="abdominalConfig", default=False, action="store_true",
                        help="Set up for ski10 config")

    parser.add_argument("--boundaryPatches", dest="boundaryPatches", default=False, action="store_true",
                        help="Create boundary patches")
    parser.add_argument("--edtRegional", dest="edtRegional", default=False, action="store_true",
                        help="Create edt regional patches")
    parser.add_argument("--specificLabels", dest="specificLabels", default=None, type=int, nargs="+",
                        help="set any specific labels for creating patches")
    parser.add_argument("--boundaryLabels", dest="boundaryLabels", default=None, type=int, nargs="+",
                        help="set any specific labels for creating patches")
    parser.add_argument("--edtLabels", dest="edtLabels", default=None, type=int, nargs="+",
                        help="Create edt regional patches")
    parser.add_argument("--clipLabelEdt", dest="clipLabelEdt", default=False, action="store_true",
                        help="Clip the label data boundary for edt according to patch size "
                             "- reflects usage during patch search")
    parser.add_argument("--minValue", dest="minValue", default=None, type=float,
                        help="Set a minimum value to include in data")
    parser.add_argument("--maxValue", dest="maxValue", default=None, type=float,
                        help="Set a maximum value to include in data")
    parser.add_argument("--addCoordinates", dest="addCoordinates", default=False, action="store_true",
                        help="Set a maximum value to include in data")
    parser.add_argument("--addDistToCentre", dest="addDistToCentre", default=False, action="store_true",
                        help="Set a maximum value to include in data")
    parser.add_argument("--addYDistToCentre", dest="addYDistToCentre", default=False, action="store_true",
                        help="Set a maximum value to include in data")
    parser.add_argument("--clipPatchSize", dest="clipPatchSize", default=None, type=int,
                        help="Create edt regional patches")
    parser.add_argument("--normaliseSpatial", dest="normaliseSpatial", default=False, action="store_true",
                        help="normalise spatial info to be between 0 and 1")

    options = parser.parse_args()

    builder = DataSetPatchMaker(options.dataPath, options.labelsPath, options.imageExpand,
                                specificLabels=options.specificLabels, boundaryLabels=options.boundaryLabels,
                                preEdtErosion=options.preEdtErosion)
    if options.ski10Config:
        edtLabels = [1, 3]
    elif options.ski10Config2:
        edtLabels = [1, 2, 3, 4]
    elif options.abdominalConfig:
        edtLabels = [3, 4, 7, 8]
    else:
        edtLabels = options.edtLabels

    print "EDT labels:", edtLabels

    auto_make_dir(options.savePath)

    if not isinstance(options.patchSize, int):
        options.patchSize = tuple(options.patchSize)
        if len(options.patchSize) == 1:
            options.patchSize = options.patchSize[0]

    print "Patch size:", options.patchSize

    if options.boundaryPatches:
        if options.edtRegional:
            if options.spatialWeight > 0:
                builder.save_edt_regional_boundary_spatial_patch_dicts(options.patchSize, options.savePath,
                                                                       options.dilation,
                                                                       options.edtLabelsPath, edtLabels=edtLabels,
                                                                       spatialWeight=options.spatialWeight,
                                                                       regionalOverlap=options.regionalOverlap,
                                                                       clipLabelEdt=options.clipLabelEdt,
                                                                       minValue=options.minValue,
                                                                       maxValue=options.maxValue,
                                                                       clipPatchSize=options.clipPatchSize)
            else:
                builder.save_edt_regional_boundary_patch_dicts(options.patchSize, options.savePath, options.dilation,
                                                               options.edtLabelsPath, edtLabels=edtLabels,
                                                               minValue=options.minValue, maxValue=options.maxValue)
        else:
            builder.save_boundary_patch_dicts(options.patchSize, options.savePath, options.dilation,
                                              minValue=options.minValue, maxValue=options.maxValue)
    else:
        if options.edtRegional:
            builder.save_edt_regional_patch_dicts(options.patchSize, options.savePath, options.edtLabelsPath,
                                                  edtLabels=edtLabels, minValue=options.minValue,
                                                  maxValue=options.maxValue)
        elif options.addCoordinates:
            builder.save_coordinate_patch_dicts(options.patchSize, options.savePath,
                                                minValue=options.minValue, maxValue=options.maxValue,
                                                normaliseSpatial=options.normaliseSpatial)
        elif options.addDistToCentre:
            print "Building with CentreDist"
            builder.save_centre_dist_patch_dicts(options.patchSize, options.savePath,
                                                 minValue=options.minValue, maxValue=options.maxValue,
                                                 normaliseSpatial=options.normaliseSpatial)
        elif options.addYDistToCentre:
            print "Building with YCentreDist"
            builder.save_y_centre_dist_patch_dicts(options.patchSize, options.savePath,
                                                   minValue=options.minValue, maxValue=options.maxValue,
                                                   normaliseSpatial=options.normaliseSpatial)
        else:
            builder.save_patch_dicts(options.patchSize, options.savePath,
                                     minValue=options.minValue, maxValue=options.maxValue)
    print "Done!"