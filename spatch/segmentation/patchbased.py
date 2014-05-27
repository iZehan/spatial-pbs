"""
Created on 12 Apr 2013

This works directly on the images without using pre-built patch dictionaries
Atlas selection does not occur here

@author: zw606
"""

from collections import deque

import numpy
from numpy.core.umath import logical_and
from spatch.image.transform import zero_out_boundary

from labelfusion import non_local_means_presq
from helper import arrange_results_by_location, DictResultsSorter

from spatch.image import patchmaker, mask
from spatch.image.mask import union_masks, get_min_max_mask
from spatch.image.patchmaker import EDT
from spatch.knn.searcher import ImagePatchSearcher
from spatch.utilities.paralleling import multi_process_list_with_consumer


__all__ = ["SAPS", "RegionalSAPS"]

# Change this to impose hard limit for memory consuming tasks
MAX_JOBS = 1


def _query_atlas(atlas, searchHandler, patches, labelsIndices, k, queryMask, boundaryDilation, boundaryClipping,
                 regionIndex, spatialRegionLabels, dtLabels, preDtErosion, dtSeeds):
    return searchHandler.query(atlas, patches, labelsIndices, k, roiMask=queryMask,
                               boundaryDilation=boundaryDilation, regionIndex=regionIndex,
                               spatialRegionLabels=spatialRegionLabels,
                               dtLabels=dtLabels, preDtErosion=preDtErosion,
                               boundaryClipping=boundaryClipping, dtSeeds=dtSeeds)


class RegionalSAPS(object):
    """
        Runs SAPS in regions - regions defined either by regionMasks or by using EDT to define regions
        atlases must be pre-selected for each region

    """

    def __init__(self, regionalAtlasDict, imagesFolder, labelsFolder, patchSize, spatialWeight=None,
                 spatialInfoType=EDT, dtLabelsFolder=None, gdtImagesFolder=None,
                 boundaryDilation=2, boundaryClipping=0, minValue=None, maxValue=None, imageExpand=True, is2D=False):
        """
        regionIndex are typically indexed by the EDT label
        @param regionalAtlasDict: Dictionary of region index to atlases {regionIndex:[atlases]}
        @param imagesFolder:
        @param labelsFolder:
        @param patchSize:
        @param spatialWeight:
        @param boundaryDilation: specify size of dilation - erosion if doing boundary refinement
        @param boundaryClipping: boundary around labels to clip in queryData (if using EDT-spatial info)
        @param spatialInfoType: what type of spatial information to use
        @param imageExpand: expand images by 1 pixel around borders (by interpolation) or not
        """
        self.regionalAtlasDict = regionalAtlasDict
        self.numRegions = len(regionalAtlasDict)
        self.saps = SAPS(imagesFolder, labelsFolder, patchSize,
                         spatialWeight=spatialWeight, minValue=minValue, maxValue=maxValue,
                         boundaryDilation=boundaryDilation, boundaryClipping=boundaryClipping,
                         spatialInfoType=spatialInfoType, dtLabelsFolder=dtLabelsFolder,
                         gdtImagesFolder=gdtImagesFolder,
                         imageExpand=imageExpand, is2D=is2D)
        self.spatialInfoType = spatialInfoType

    def set_atlas_dictionary(self, regionalAtlasDict):
        self.regionalAtlasDict = regionalAtlasDict

    def label_region(self, regionMask, regionIndex, imageData, k, queryMaskDict=None, spatialInfo=None, dtLabels=None,
                     preEdtErosion=0, useQueryMaskOnAtlases=False, dtSeeds=None, overallMask=None,
                     spatialRegionLabels=None, numProcessors=8):
        if queryMaskDict is not None:
            newQueryDict = dict((label, logical_and(queryMaskDict[label], regionMask)) for label in queryMaskDict)
        else:
            newQueryDict = {1: regionMask}

        return self.saps.label_image(imageData, k, self.regionalAtlasDict[regionIndex],
                                     queryMaskDict=newQueryDict, spatialInfo=spatialInfo, dtLabels=dtLabels,
                                     preDtErosion=preEdtErosion, spatialRegionIndex=regionIndex,
                                     useQueryMaskOnAtlases=useQueryMaskOnAtlases, dtSeeds=dtSeeds,
                                     overallMask=overallMask, spatialRegionLabels=spatialRegionLabels,
                                     numProcessors=numProcessors)

    def label_image(self, imageData, k, queryMaskDict=None, regionalMaskDict=None, spatialInfo=None, dtLabels=None,
                    preDtErosion=0, useQueryMaskOnAtlases=False, dtSeeds=None, overallMask=None, numProcessors=8):
        """
        @param regionalMaskDict: dictionary of masks {regionIndex : mask}
        @param imageData: numpy array representing image
        @param k:
        @param queryMaskDict: dictionary of (label: mask) indicating which labels can be queried for at each location
                including background label. The union of masks defines total query mask
                - if None is given, all labels will be searched for at all locations
                - if a dictionary with a single (label, mask) is given, all labels will be search for within
                the mask
                - assume masks are same shape as image Data
        @param regionalMaskDict: dictionary of {regionIndex: mask} indicating query mask for each region
        @param spatialInfo: list or array of 3D spatial info (assume it is unweighted)
        @param dtLabels:
        @param preDtErosion:
        @param useQueryMaskOnAtlases:
        @param numProcessors:
        """
        if len(regionalMaskDict) != self.numRegions:
            if len(regionalMaskDict) > self.numRegions:
                raise Exception("Too many region masks defined!")
            else:
                raise Exception("Not enough region masks defined!")

        if spatialInfo is None and self.spatialInfoType in patchmaker.GENERIC_SPATIAL_INFO_TYPES:
            spatialInfo = patchmaker.get_generic_spatial_info(imageData, self.spatialInfoType)

        partResults = numpy.zeros(imageData.shape, numpy.uint8)
        regionCount = 1
        for region in regionalMaskDict:
            print "[INFO]--->Labelling region", regionCount, "out of", len(regionalMaskDict)
            if numpy.count_nonzero(regionalMaskDict[region]) > 0:
                partResults += self.label_region(regionalMaskDict[region], region, imageData, k,
                                                 queryMaskDict=queryMaskDict,
                                                 spatialInfo=spatialInfo, dtLabels=dtLabels,
                                                 preEdtErosion=preDtErosion,
                                                 useQueryMaskOnAtlases=useQueryMaskOnAtlases,
                                                 dtSeeds=dtSeeds,
                                                 overallMask=overallMask,
                                                 spatialRegionLabels=regionalMaskDict.keys(),
                                                 numProcessors=numProcessors)
            regionCount += 1
        return partResults

    def set_images_labels_paths(self, imagePaths, labelsPath, dtLabelsPath=None):
        self.saps.set_images_labels_paths(imagePaths, labelsPath, dtLabelsPath=dtLabelsPath)


class SAPS(object):
    """
        Runs spatially aware patch based segmentation, searching trees to retrive kNN from initially provided atlases 
        and then performs non-local means label fusion to derive label for each voxel
        This does not perform atlas selection
        
    """

    def __init__(self, imagesFolder, labelsFolder, patchSize, spatialWeight=None, spatialInfoType=None,
                 dtLabelsFolder=None, gdtImagesFolder=None, boundaryDilation=2, boundaryClipping=0,
                 minValue=None, maxValue=None, imageExpand=True, is2D=False, rescaleIntensities=False):
        """
        @param imagesFolder:
        @param labelsFolder:
        @param patchSize:
        @param spatialWeight:
        @param boundaryDilation: specify size of dilation - erosion if doing boundary refinement
        @param boundaryClipping: boundary around labels to clip in queryData (if using EDT-spatial info)
        @param spatialInfoType: type of spatial information to use
        @param imageExpand: expand atlas images by 1 voxel around borders (by interpolation) or not
        """

        imageSearcher = ImagePatchSearcher
        self.searchHandler = imageSearcher(imagesFolder, labelsFolder, patchSize,
                                           spatialWeight=spatialWeight, spatialInfoType=spatialInfoType,
                                           minValue=minValue, maxValue=maxValue, imageExpand=imageExpand,
                                           dtLabelsFolder=dtLabelsFolder, gdtImagesFolder=gdtImagesFolder,
                                           is2D=is2D,
                                           rescaleIntensities=rescaleIntensities)
        self.is2D = is2D
        if isinstance(patchSize, int):
            patchSize = [patchSize] * 3
        self.patchSize = tuple(patchSize)
        self.spatialWeight = spatialWeight
        self.spatialInfoType = spatialInfoType
        self.minValue = minValue
        self.maxValue = maxValue
        self.boundaryDilation = boundaryDilation
        self.boundaryClipping = boundaryClipping

    def label_image(self, imageData, k, atlases, queryMaskDict=None, spatialInfo=None, dtLabels=None,
                    preDtErosion=0, spatialRegionIndex=None, useQueryMaskOnAtlases=False,
                    dtSeeds=None, overallMask=None, spatialRegionLabels=None, numProcessors=8):
        """
        @return results: labels of same shape/size as imageData: note if imageExpand, will need to modify

        @param imageData: image data to query
        @param k: number of nearest neighbours to search from atlases
        @param atlases: atlases (file names) to use for labelling
        @param queryMaskDict: dictionary of (label, mask) indicating which labels can be queried for at each location
                including background label. The union of masks defines total query mask
                - if None is given, all labels will be searched for at all locations
                - if a dictionary with a single (label, mask) is given, all labels will be search for within
                the mask
                - assume masks are same shape as image Data
        @param spatialInfo: list or array of 3D spatial info (assume it is unweighted)
        @param dtLabels: labels to get DT-spatial info for in atlases
        @param preDtErosion: specify any erosion of labels before atlases
        @param spatialRegionIndex: index (dt label) when using dts to define region (valid if not useQueryMaskOnAtlases)
        @param useQueryMaskOnAtlases: apply the queryMask to atlases as well as target image
                                        (regions no longer defined by edtRegionIndex) - requires image registration
        @param numProcessors: number of processors to use

        """

        print "[INFO]--->Number of atlases to use:", len(atlases)
        print "[INFO]--->SpatialWeight:", self.spatialWeight
        locationModifier = numpy.int16(numpy.asarray(self.patchSize) / 2)

        print "[INFO]--->TargetImage Shape:", imageData.shape

        # setup spatial info
        if self.spatialInfoType is None or self.spatialWeight is None or self.spatialWeight == 0:
            spatialInfo = None
        else:
            if spatialInfo is None:
                if self.spatialInfoType in patchmaker.GENERIC_SPATIAL_INFO_TYPES:
                    print "Getting initial spatial information..."
                    spatialInfo = patchmaker.get_generic_spatial_info(imageData, self.spatialInfoType)
                elif self.spatialInfoType in patchmaker.SPATIAL_INFO_TYPES:
                    raise Exception("No spatial info provided")
            spatialInfo = numpy.asarray(spatialInfo, numpy.float) * self.spatialWeight

        # initialise masks, indices
        queryMask = None
        patchesMask = get_min_max_mask(imageData, minValue=self.minValue, maxValue=self.maxValue)
        labelsIndices = None
        if queryMaskDict is not None:
            if patchesMask is not None:
                patchesMask = logical_and(union_masks(queryMaskDict.values()), patchesMask)
            else:
                patchesMask = union_masks(queryMaskDict.values())

            # cannot query boundary of image if not big enough for a patch
            patchesMask = zero_out_boundary(patchesMask, locationModifier)
            locationList = numpy.atleast_2d(numpy.squeeze(numpy.transpose(numpy.nonzero(patchesMask))))
            locationList = list(locationList)
            if useQueryMaskOnAtlases:
                queryMask = patchesMask
                spatialRegionIndex = None
                if overallMask is not None and self.boundaryDilation > 0:
                    queryMask = mask.dilate_mask(queryMask, self.boundaryDilation)
                    queryMask = logical_and(overallMask, queryMask)

            # get labelIndices
            if len(queryMaskDict.keys()) > 1:
                labelsIndices = dict()
                for i in xrange(len(locationList)):
                    for label in queryMaskDict:
                        if queryMaskDict[label][tuple(locationList[i])]:
                            try:
                                labelsIndices[label].append(i)
                            except KeyError:
                                labelsIndices[label] = deque([i])
                for label in labelsIndices:
                    labelsIndices[label] = numpy.asarray(labelsIndices[label])
        else:
            if patchesMask is not None:
                patchesMask = zero_out_boundary(patchesMask, locationModifier)
            else:
                patchesMask = zero_out_boundary(numpy.ones(imageData.shape, numpy.bool), locationModifier)
            locationList = numpy.atleast_2d(numpy.squeeze(numpy.transpose(numpy.nonzero(patchesMask))))
            locationList = list(locationList)

        numQueryLocations = len(locationList)

        print "Num Query Locations:", numQueryLocations
        # create patches
        print "Creating patches..."
        patches = patchmaker.get_patches(imageData, self.patchSize, maskData=patchesMask, spatialData=spatialInfo)

        consumerObj = DictResultsSorter(k, non_local_means_presq, labelsIndices, numQueryLocations)

        print "Searching..."
        results = multi_process_list_with_consumer(atlases, self.query_worker, consumerObj,
                                                   min(MAX_JOBS, numProcessors),
                                                   patches, labelsIndices, k, queryMask, self.boundaryDilation,
                                                   spatialRegionIndex, spatialRegionLabels, dtLabels, preDtErosion,
                                                   dtSeeds)

        print "Processing distances..."
        # results = consumerObj.results()
        results = arrange_results_by_location(results, locationList, imageData.shape)
        # Remove and clear loaded trees to clear memory

        return results

    def query_worker(self, atlasList, patches, labelsIndices, k, queryMask, boundaryDilation, regionIndex,
                     spatialRegionLabels, dtLabels, preDtErosion, dtSeeds, queueOut):
        put = queueOut.put
        query = self.searchHandler.query
        for atlas in atlasList:
            put(query(atlas, patches, labelsIndices, k, roiMask=queryMask,
                      boundaryDilation=boundaryDilation, regionIndex=regionIndex,
                      spatialRegionLabels=spatialRegionLabels,
                      dtLabels=dtLabels, preDtErosion=preDtErosion,
                      boundaryClipping=self.boundaryClipping, dtSeeds=dtSeeds))

    def _query_atlas(self, atlas, patches, labelsIndices, k, queryMask, boundaryDilation, regionIndex,
                     spatialRegionLabels, dtLabels, preDtErosion, dtSeeds, consumerObj):
        results = self.searchHandler.query(atlas, patches, labelsIndices, k, roiMask=queryMask,
                                           boundaryDilation=boundaryDilation, regionIndex=regionIndex,
                                           spatialRegionLabels=spatialRegionLabels,
                                           dtLabels=dtLabels, preDtErosion=preDtErosion,
                                           boundaryClipping=self.boundaryClipping, dtSeeds=dtSeeds)
        consumerObj.process(results)

    def set_images_labels_paths(self, imagePaths, labelsPath, dtLabelsPath=None):
        self.searchHandler.set_images_labels_paths(imagePaths, labelsPath, dtLabelsPath=dtLabelsPath)
