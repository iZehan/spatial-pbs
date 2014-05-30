"""
Created on 6 Dec 2012

@author: Zehan Wang
"""
from __future__ import division
import os
import math
from os.path import join

import numpy
from sklearn.neighbors import BallTree as ScikitBallTree

from balltree import BallTree

from spatch.image.patchmaker import PatchMaker, EDT


#import pyflann

CUSTOM_BALL_TREE = "custom-ball-tree"
SK_DUAL_TREE = "scikit-ball-tree-dual"

__all__ = ["ImagePatchSearcher"]


def choose_leaf_size(numItems):
    """
    Rough choice for selecting leaf size - performance will vary
    """
    multiplier = 14
    if numItems < 100:
        multiplier = 1
    elif numItems < 1000:
        multiplier = 2
    elif numItems < 3000:
        multiplier = 3
    elif numItems < 5000:
        multiplier = 4
    elif numItems < 10000:
        multiplier = 5
    elif numItems < 20000:
        multiplier = 6
    elif numItems < 32768:
        multiplier = 7
    elif numItems < 524288:
        multiplier = 8
    elif numItems < 1048576:
        multiplier = 9
    elif numItems < 2097152:
        multiplier = 10
    elif numItems < 4194304:
        multiplier = 12
    elif numItems < 16777216:
        multiplier = 24
    elif numItems < 33554432:
        multiplier = 26
    elif numItems < 134217728:
        multiplier = 28
    return max(1, int(multiplier * math.log(numItems, 2)))


def build_ball_tree(data):
    return BallTree(numpy.asarray(data), choose_leaf_size(len(data)))


def query_ball_tree(tree, queryData, k):
    return tree.query_many(queryData, k)


def build_scikit_ball_tree(data):
    return ScikitBallTree(numpy.asarray(data), choose_leaf_size(len(data)))


def query_dual_tree(tree, queryData, k):
    """
    @param tree: must be a sklearn BallTree or KDTree
    @return: k nearest neighbours
    """
    distances, indices = tree.query(queryData, k, dualtree=True)
    # square results since we're working with squared euclidean distances for the label fusion
    return distances ** 2, indices


class KnnSearcher(object):
    def __init__(self, knnStructureType, maxData=None):
        self.maxData = maxData
        if knnStructureType == CUSTOM_BALL_TREE:
            self.build_method = build_ball_tree
            self.query_method = query_ball_tree
        if knnStructureType == SK_DUAL_TREE:
            self.build_method = build_scikit_ball_tree
            self.query_method = query_dual_tree

    def build_and_query(self, data, queryData, k):
        if self.maxData is not None and len(data) > self.maxData:
            print "Number of data items:", len(data)
            print " - Exceeded maximum data size - random sampling to", self.maxData
            data = data[numpy.random.choice(len(data), self.maxData, replace=False)]
        structure = self.build_method(data)
        return self.query_method(structure, queryData, k)


class ImagePatchSearcher(object):
    def __init__(self, imagesFolder, labelsFolder, patchSize, spatialWeight=None, minValue=None, maxValue=None,
                 spatialInfoType=EDT, imageExpand=True, dtLabelsFolder=None, gdtImagesFolder=None,
                 is2D=False, rescaleIntensities=False, knnStructureType=CUSTOM_BALL_TREE):
        self.set_images_labels_paths(imagesFolder, labelsFolder, dtLabelsFolder, gdtImagesFolder)
        self.patchSize = patchSize
        self.spatialWeight = spatialWeight
        self.spatialInfoType = spatialInfoType
        self.minValue = minValue
        self.maxValue = maxValue
        self.imageExpand = imageExpand
        self.check_resources_exist()
        self.is2D = is2D
        self.rescaleIntensities = rescaleIntensities
        self.knnSearcher = KnnSearcher(knnStructureType)

    def set_images_labels_paths(self, imagesFolder, labelsFolder, dtLabelsPath=None, gdtImagesFolder=None):
        self.imagesFolder = imagesFolder
        self.labelsFolder = labelsFolder
        self.dtLabelsFolder = dtLabelsPath
        self.gdtImagesFolder = gdtImagesFolder
        self.check_resources_exist()

    def check_resources_exist(self):
        if not os.path.exists(self.imagesFolder):
            raise Exception("Image folder does not exists: " + self.imagesFolder)
        if not os.path.exists(self.labelsFolder):
            raise Exception("Labels folder does not exists: " + self.labelsFolder)
        if self.dtLabelsFolder is not None and not os.path.exists(self.dtLabelsFolder):
            raise Exception("EDT Labels folder does not exists: " + self.dtLabelsFolder)
        if self.gdtImagesFolder is not None and not os.path.exists(self.gdtImagesFolder):
            raise Exception("EDT Labels folder does not exists: " + self.gdtImagesFolder)

    def query(self, atlas, patches, labelsIndices, k, roiMask=None, boundaryDilation=None,
              regionIndex=None, spatialRegionLabels=None,
              dtLabels=None, preDtErosion=0, boundaryClipping=0, dtSeeds=None, **kwargs):
        """

        @param spatialRegionLabels: labels to determine the which labelled structures to base regions around
        @param dtSeeds:
        @param kwargs:
        @param atlas: filename of image to search
        @param patches: list/array of patches
        @param labelsIndices: dict(label, index in patches)
        @param k: number of nearest neighbours
        @param roiMask: mask to limit query region
        @param boundaryDilation: boundary dilation in atlases labels to search, if None, search whole atlas
        @param regionIndex: region index - if using edtLabels to split image into regions (index = edt label)
        @param dtLabels: labels used to construct edt spatial info
        @param preDtErosion:
        @param boundaryClipping: clip boundaries before getting EDT can be int or collections.Iterable

        @return dictionary of {label: distances} for the knn of each label
        """
        # load data and create patches and tree
        dtLabelsPath = None
        gdtImagePath = None
        if self.dtLabelsFolder is not None:
            dtLabelsPath = join(self.dtLabelsFolder, atlas)
        if self.gdtImagesFolder is not None:
            gdtImagePath = join(self.gdtImagesFolder, atlas)

        patchMaker = PatchMaker(join(self.imagesFolder, atlas), join(self.labelsFolder, atlas), self.imageExpand,
                                minValue=self.minValue, maxValue=self.maxValue, dtLabelsPath=dtLabelsPath,
                                gdtImagePath=gdtImagePath,
                                is2D=self.is2D, rescaleIntensities=self.rescaleIntensities)
        patchDict = patchMaker.get_patch_dict(self.patchSize,
                                              spatialWeight=self.spatialWeight, boundaryDilation=boundaryDilation,
                                              spatialRegionIndex=regionIndex, spatialRegionLabels=spatialRegionLabels,
                                              spatialLabels=dtLabels,
                                              labelErosion=preDtErosion, boundaryClipSize=boundaryClipping,
                                              spatialInfoType=self.spatialInfoType, roiMask=roiMask, dtSeeds=dtSeeds,
                                              includePatchSizeKey=False)

        if patchDict is None:
            return None

        return self.__query_patches(patchDict, patches, labelsIndices, k)

    def __query_patches(self, atlasPatchDict, queryPatches, queryLabelsIndices, k, **kwargs):
        """
            runs knn queries on patches, returns dictionary of distances for each label
        """
        k = numpy.int32(k)
        if queryLabelsIndices is None:
            return dict((label, self.knnSearcher.build_and_query(atlasPatchDict[label], queryPatches, k)[0])
                        for label in atlasPatchDict if len(atlasPatchDict[label]) > 0)
        else:
            results = dict()
            for label in atlasPatchDict:
                if label in queryLabelsIndices and len(atlasPatchDict[label]) > 0:
                    try:
                        results[label] = self.knnSearcher.build_and_query(atlasPatchDict[label],
                                                                          queryPatches[queryLabelsIndices[label]], k)[0]
                    except IndexError:
                        results[label] = [[float("inf")] for _ in xrange(len(queryLabelsIndices[label]))]
                    except:
                        print "Tried to build ball tree with the following data:"
                        print atlasPatchDict[label]
                        print "Shape:", atlasPatchDict[label].shape
                        print "QueryData shape:", queryPatches.shape
                        print "queryLabelsIndices[label].shape", queryLabelsIndices[label].shape
                        raise
        return results


# def random_sample(dataset, percentage=0.5):
#     totalItems = dataset.shape[0]
#     numItems = int(round(totalItems * percentage))
#     return dataset[numpy.random.choice(totalItems, numItems, replace=False)]

# class FlannLookUp(PatchDictionaryLookUp):
#     """
#         performs block queries on one atlas at a time rather than all atlases
#         resourcesFolder is a tuple of (patchesDictFolder, flannFolder)
#     """
# 
#     def check_resources_exist(self, files):
#         for x in files:
#             if not os.path.exists(self.resourceFolders[0] + x + ".dict"):
#                 raise Exception("Resource does not exists:" + self.resourceFolders[0] + x + ".dict")
#             if len(glob.glob(self.resourceFolders[1] + x + "_*.fla")) == 0:
#                 raise Exception("Resource does not exists:" + self.resourceFolders[1] + x + "_*.fla")
#     
#     def query(self, atlas, patches, labelsIndices, k):
#         """
#             patches is a numpy array of vectorized patches (2D numpy array)
#             labelsIndices is a dictionary of {label: numpy array([indices in patches])}
#               - can be None if same for all labels
#         """
#         #Note: FLANN returns squared distances! must bear this in mind when using distances for atlas selection
#         results = dict()
#         with open(self.resourceFolders[0] +"/" + atlas + ".dict", "r") as data:
#             tempDict = pickle.load(data)
#         for label in tempDict:
#             data = modify_spatial_weight(numpy.asarray(tempDict[label]), self.spatialMultiplier, self.numSpatialDims)
#             flann = pyflann.FLANN()
#             flann.load_index(self.resourceFolders[1] + atlas + "_" + str(label) + ".fla", data)
#             try:
#                 results[label] = self.query_flann(flann, patches[labelsIndices[label]], k)
#             except TypeError:
#                 results[label] = self.query_flann(flann, patches, k)
#         return results
#     
#     def query_flann(self, flann, queryPatches, k):
#         try: 
#             return flann.nn_index(queryPatches, k)[1]
#         except AssertionError:
#             return flann.nn_index(queryPatches, len(flann._FLANN__curindex_data))[1]
