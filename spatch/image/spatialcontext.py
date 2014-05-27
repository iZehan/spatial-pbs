"""
Created on 13/01/14

@author: zw606
"""

from sklearn.externals.joblib import Parallel, delayed
import numpy
from numpy.core.umath import logical_not
from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude, center_of_mass
from utilities.misc import auto_non_background_labels

from gdt import geodesic_distance_transform
from mask import erode_mask, get_bounding_box, dilate_mask
from transform import interpolate_to_shape, resample_data_to_shape, zero_out_boundary, image_boundary_expand


EDT = "edt"
GDT = "gdt"
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


__all__ = ["edt", "gdt", "eroded_gdt", "eroded_gdt", "multi_label_edt", "multi_label_edt_dict",
           "multi_label_gdt", "multi_label_gdt_dict", "get_generic_spatial_info", "get_coordinates",
           "get_coordinates_from_data_shape", "get_dist_to_reference_point"]


def edt(binaryData, voxelSpacing=1, subtractInside=True, boundingMask=None):
    """
    Uses Scipy's distance_transfrom_edt - for some reason this works backwards the runs where the binary mask is 1
    @param boundingMask is a mask provided where its bounding box will be used to restrict
           where the distance transform is run for performance gain
    """
    if numpy.count_nonzero(binaryData) == 0:
        print "[WARN]--->No binary data provided to EDT"
        distanceMap = numpy.zeros(binaryData.shape)
        distanceMap[:] = numpy.NaN
        return distanceMap

    minBounds, maxBounds = None, None
    if boundingMask is not None:
        minBounds, maxBounds = get_bounding_box(boundingMask)
        binaryData = binaryData[minBounds[0]:maxBounds[0], minBounds[1]:maxBounds[1], minBounds[2]:maxBounds[2]]

    distanceMap = distance_transform_edt(logical_not(binaryData), sampling=voxelSpacing)
    if subtractInside:
        distanceMap -= distance_transform_edt(binaryData, sampling=voxelSpacing)

    if boundingMask is not None:
        temp = numpy.zeros(boundingMask.shape)
        temp[minBounds[0]:maxBounds[0], minBounds[1]:maxBounds[1], minBounds[2]:maxBounds[2]] = distanceMap
        distanceMap = temp

    return distanceMap


def eroded_edt(binaryData, erosion, voxelSpacing=1, is2D=False, subtractInside=True, boundingMask=None):
    """
        returns edt results for labels using scipy.ndimage.morphology.distance_transform_edt instead of irtk
    """
    tempData = erode_mask(binaryData, erosion, is2D=is2D)
    return edt(tempData, voxelSpacing=voxelSpacing, subtractInside=subtractInside, boundingMask=boundingMask)


def multi_label_edt(labelData, erosion=0, voxelSpacing=1, specificLabels=None, imageBoundaryClipping=0,
                    is2D=False, subtractInside=True, boundingMask=None):
    """
        Use by default instead of IRTK edt since this does not require IO or a temporary save location
        returns edt results for labels using scipy.ndimage.morphology.distance_transform_edt instead of irtk
    """
    labelData = zero_out_boundary(labelData, imageBoundaryClipping)

    if specificLabels is None:
        specificLabels = auto_non_background_labels(labelData)
    distanceMaps = [eroded_edt(labelData == label, erosion, voxelSpacing=voxelSpacing,
                               is2D=is2D, subtractInside=subtractInside, boundingMask=boundingMask)
                    for label in specificLabels]

    return distanceMaps


def multi_label_edt_dict(labelData, specificLabels=None, erosion=0, imageBoundaryClipping=0, voxelSpacing=1,
                         is2D=False, subtractInside=True, boundingMask=None):
    """
    returns dictionary of (label, edt results) using scipy.ndimage.morphology.distance_transform_edt instead of irtk
    Assumes data has isotropic voxel size
    """
    if specificLabels is None:
        specificLabels = auto_non_background_labels(labelData)

    distanceMaps = multi_label_edt(labelData, erosion=erosion, voxelSpacing=voxelSpacing, specificLabels=specificLabels,
                                   imageBoundaryClipping=imageBoundaryClipping,
                                   is2D=is2D, subtractInside=subtractInside, boundingMask=boundingMask)

    return dict((specificLabels[i], distanceMaps[i]) for i in range(len(specificLabels)))


def gdt(binaryImage, image, voxelSpacing=(1.0, 1.0, 1.0), subtractInside=True, includeEDT=True,
        gamma=32, numIterations=3, boundingMask=None, magnitudes=None):
    """
        Runs geodesic distance transform based on gradient magnitues as the geodesic cost
        @param boundingMask is a mask provided where its bounding box will be used to restrict
               where the distance transform is run for performance gain
    """
    if numpy.count_nonzero(binaryImage) == 0:
        print "[WARN]--->No binary data provided to GDT"
        distanceMap = numpy.zeros(binaryImage.shape)
        distanceMap[:] = numpy.NaN
        return distanceMap

    if magnitudes is None:
        image = numpy.asarray(image, dtype=numpy.float64)
        magnitudes = gaussian_gradient_magnitude(image, 0.5)
        if includeEDT and gamma != 1:
            magnitudes *= gamma
    minBounds, maxBounds = None, None
    if boundingMask is not None:
        minBounds, maxBounds = get_bounding_box(boundingMask)
        binaryImage = binaryImage[minBounds[0]:maxBounds[0], minBounds[1]:maxBounds[1], minBounds[2]:maxBounds[2]]
        magnitudes = magnitudes[minBounds[0]:maxBounds[0], minBounds[1]:maxBounds[1], minBounds[2]:maxBounds[2]]

    distanceMap = geodesic_distance_transform(binaryImage, magnitudes,
                                              numIterations=numIterations, spacing=voxelSpacing, includeEDT=includeEDT)
    if subtractInside:
        distanceMap -= geodesic_distance_transform(logical_not(binaryImage), magnitudes,
                                                   numIterations=numIterations, spacing=voxelSpacing,
                                                   includeEDT=includeEDT)
    if magnitudes is None and includeEDT and gamma != 1:
        # make it more comparable when different gamma are used
        distanceMap /= gamma

    if boundingMask is not None:
        temp = numpy.zeros(boundingMask.shape)
        temp[minBounds[0]:maxBounds[0], minBounds[1]:maxBounds[1], minBounds[2]:maxBounds[2]] = distanceMap
        distanceMap = temp

    return distanceMap


def eroded_gdt(binaryImage, erosion, image, gamma=32, numIterations=3, spacing=(1.0, 1.0, 1.0),
               subtractInside=True, includeEDT=True, is2D=False, boundingMask=None, magnitudes=None):
    binaryImage = erode_mask(binaryImage, erosion=erosion, is2D=is2D)
    return gdt(binaryImage, image, gamma=gamma, numIterations=numIterations, voxelSpacing=spacing,
               subtractInside=subtractInside, includeEDT=includeEDT, boundingMask=boundingMask,
               magnitudes=magnitudes)


def multi_label_gdt(labelsData, imageData, specificLabels=None, gamma=32,
                    erosion=0, imageBoundaryClipping=0, voxelSize=(1., 1., 1.),
                    subtractInside=True, includeEDT=True, is2D=False, boundingMask=None, makeIsotropic=False,
                    numJobs=None):
    """
    handles if labelsData.shape != imageData.shape by interpolating
    """
    imageData = numpy.asarray(imageData, dtype=numpy.float64)
    # make a copy so as not to change original label data
    replacementValue = 0
    if labelsData.min() == -1:
        replacementValue = -1

    labelsData = zero_out_boundary(labelsData.copy(), imageBoundaryClipping, replacementValue=replacementValue)
    voxelSize = numpy.asarray(voxelSize, dtype=numpy.float64)
    if makeIsotropic:
        if not numpy.all(voxelSize == voxelSize[0]):
            #make isotropic
            minVSize = min(voxelSize)
            multipliers = voxelSize / minVSize
            newShape = numpy.asarray(imageData.shape) * multipliers
            newShape = tuple(numpy.int16(numpy.round(newShape)))
            # print "making image isotropic...", imageData.shape, "to", newShape
            imageData = interpolate_to_shape(imageData, newShape, interpolationType="cubic")
            voxelSize = (minVSize, minVSize, minVSize)

    originalDataShape = None
    if boundingMask is not None and boundingMask != imageData:
        boundingMask = interpolate_to_shape(boundingMask, imageData.shape)

    if labelsData.shape != imageData.shape:
        originalDataShape = labelsData.shape
        labelsData = interpolate_to_shape(labelsData, imageData.shape, interpolationType="NN")

    if specificLabels is None:
        specificLabels = auto_non_background_labels(labelsData)

    if numJobs is None:
        numJobs = len(specificLabels)

    gradientMagnitudes = gaussian_gradient_magnitude(imageData, 0.5)

    if includeEDT and gamma != 1:
        gradientMagnitudes *= gamma

    # handle background label different if its included
    distanceMaps = []
    if 0 in specificLabels:
        binaryImage = labelsData != 0
        binaryImage = dilate_mask(binaryImage, dilation=erosion, is2D=is2D)
        binaryImage = logical_not(binaryImage)
        distMap = gdt(binaryImage, imageData, subtractInside=subtractInside, includeEDT=includeEDT,
                      boundingMask=boundingMask, magnitudes=gradientMagnitudes)
        distanceMaps = [distMap]

    distanceMaps += Parallel(numJobs)(
        delayed(eroded_gdt)(labelsData == label, erosion, imageData,
                            spacing=voxelSize, boundingMask=boundingMask,
                            subtractInside=subtractInside, includeEDT=includeEDT,
                            is2D=is2D, magnitudes=gradientMagnitudes)
        for label in specificLabels if label != 0)

    if originalDataShape is not None:
        distanceMaps = Parallel(numJobs)(
            delayed(resample_data_to_shape)(distanceMaps[i], originalDataShape, interpolationType="linear")
            for i in range(len(distanceMaps)))

    distanceMaps = numpy.asarray(distanceMaps)
    if includeEDT and gamma != 1:
        distanceMaps /= gamma

    return distanceMaps


def multi_label_gdt_dict(labelsData, imageData, specificLabels=None, gamma=32,
                         erosion=0, imageBoundaryClipping=0, voxelSize=(1., 1., 1.),
                         subtractInside=True, includeEDT=True, is2D=False, boundingMask=None):
    labelDistances = multi_label_gdt(labelsData, imageData, specificLabels=specificLabels, gamma=gamma,
                                     erosion=erosion, imageBoundaryClipping=imageBoundaryClipping, voxelSize=voxelSize,
                                     subtractInside=subtractInside, includeEDT=includeEDT, is2D=is2D,
                                     boundingMask=boundingMask)
    labelDistanceDictionary = dict()
    for i in range(len(specificLabels)):
        labelDistanceDictionary[specificLabels[i]] = labelDistances[i]
    return labelDistanceDictionary


def region_dict_from_dt_dict(dtResultsDict, regionalOverlap=None, specificRegionIndex=None, is2D=False):
    """
    returns a dictionary of {regionIndex, mask} according to edt separation
    """
    regionIndices = dtResultsDict.keys()
    nearestRegions = numpy.argmin(numpy.asarray(dtResultsDict.values()), axis=0)
    if specificRegionIndex is None:
        regionalMaskDict = dict((regionIndices[i], nearestRegions == i) for i in xrange(len(regionIndices)))
        if regionalOverlap:
            for index in regionalMaskDict:
                regionalMaskDict[index] = dilate_mask(regionalMaskDict[index], regionalOverlap, is2D=is2D)
        return regionalMaskDict
    else:
        regionalMask = nearestRegions == regionIndices.index(specificRegionIndex)
        if regionalOverlap:
            regionalMask = dilate_mask(regionalMask, regionalOverlap, is2D=is2D)
        return regionalMask


def regions_from_dt_results(edtResults, regionalOverlap=0, is2D=False):
    """
        a list of region masks according to distance transform separation
    """
    nearestRegions = numpy.argmin(edtResults, axis=0)
    regions = [nearestRegions == i for i in xrange(len(edtResults))]
    if regionalOverlap > 0:
        regions = [dilate_mask(region, regionalOverlap, is2D=is2D) for region in regions]
    return regions


def get_dt_spatial_context_dict(labelsData, spatialInfoType, spatialLabels=None, voxelSize=1, labelErosion=0,
                                boundaryClipSize=0, imageData=None, is2D=False, boundingMask=None, imageExpand=False):
    if spatialInfoType is None:
        return None

    spatialLabelDict = None
    if spatialInfoType == EDT:
        spatialLabelDict = multi_label_edt_dict(labelsData, specificLabels=spatialLabels,
                                                erosion=labelErosion,
                                                imageBoundaryClipping=boundaryClipSize,
                                                voxelSpacing=voxelSize, is2D=is2D, boundingMask=boundingMask)
    elif spatialInfoType == GDT:
        spatialLabelDict = multi_label_gdt_dict(labelsData, imageData, specificLabels=spatialLabels,
                                                erosion=labelErosion,
                                                imageBoundaryClipping=boundaryClipSize,
                                                voxelSize=voxelSize, is2D=is2D, boundingMask=boundingMask)
    if imageExpand:
        for label in spatialLabelDict:
            spatialLabelDict[label] = image_boundary_expand(spatialLabelDict[label], useGradient=False, is2D=is2D)

    return spatialLabelDict


def get_generic_spatial_info(imageData, spatialInfoType):
    selector = {COORDINATES: get_coordinates,
                COORDINATES_2D: get_coordinates_2d,
                DIST_CENTRE: get_dist_to_reference_point,
                NORMALISED_DIST_CENTRE: get_normalised_centre_dist,
                NORMALISED_COORDINATES: get_normalised_coordinates,
                NORMALISED_COORDINATES_2D: get_normalised_coordinates_2d,
                NORMALISED_COORDINATES_CENTRE_MASS: get_normalised_coordinates_center_mass,
                NORMALISED_DIST_CENTRE_MASS: get_normalised_distance_center_mass}
    return selector[spatialInfoType](imageData)


def get_coordinates_from_data_shape(dataShape, originPoint=None, normalise=False):
    spatialData = coordinate_array_by_shape(dataShape)

    if originPoint is not None:
        spatialData = numpy.asarray(spatialData, numpy.float)
        for i in xrange(len(dataShape)):
            if dataShape[i] > 1:
                spatialData[i] -= originPoint[i]

    if normalise:
        spatialData = numpy.asarray(spatialData, numpy.float)
        for i in xrange(len(spatialData)):
            if dataShape[i] > 1:
            # normalise to be in range [0, 1]
                spatialData[i] /= (dataShape[i] - 1)
    return spatialData


def get_coordinates_2d(imageData):
    """
    @param imageData: the image data as numpy array
    @return:
    """
    return get_coordinates(imageData)[0:2]


def get_normalised_coordinates_2d(imageData):
    return get_coordinates(imageData, normalise=True)[0:2]


def get_normalised_coordinates_center_mass(imageData):
    return get_coordinates(imageData, originPoint=center_of_mass(imageData), normalise=True)


def get_normalised_distance_center_mass(imageData):
    return get_dist_to_reference_point(imageData, referencePoint=center_of_mass(imageData), normalise=True)


def get_coordinates(imageData, originPoint=None, normalise=False):
    """
    @param imageData: the image data as numpy array
    @param originPoint: point of origin for coordinates, if None, (0, 0, 0) will be point of origin
    @param normalise: normalise coordinates to range [0, 1] or not
    @return:
    """
    return get_coordinates_from_data_shape(imageData.shape, originPoint=originPoint, normalise=normalise)


def coordinate_array_by_shape(shape):
    return numpy.indices(shape, dtype=numpy.int16)


def get_normalised_coordinates(imageData):
    return get_coordinates(imageData, normalise=True)


def get_normalised_centre_dist(imageData):
    return get_dist_to_reference_point(imageData, normalise=True)


def get_dist_to_reference_point(imageData, referencePoint=None, normalise=False):
    """
    @param imageData: the image data as numpy array
    @param referencePoint: point of reference distance to be taken from, if None, the centre of the image will be used
    @param normalise: normalise coordinates to range [0, 1] or not
    @return:
    """
    shape = numpy.asarray(imageData.shape)
    if referencePoint is None:
        # use center point, need to do -1 because shape coordinates counts from 0, but shape counts from 1
        referencePoint = (shape - 1) / 2
    coordinates = numpy.asarray(get_coordinates(imageData, originPoint=referencePoint), numpy.float)
    distances = numpy.asarray([numpy.zeros(shape, numpy.float)])
    for x in xrange(shape[0]):
        for y in xrange(shape[1]):
            for z in xrange(shape[2]):
                distances[..., x, y, z] = numpy.linalg.norm(coordinates[..., x, y, z])

    if normalise:
        distances /= numpy.max(distances)
    return distances


def dist_to_centre_spatial_info(imageData, normalise=False):
    """returns array of distances to centre"""
    return get_dist_to_reference_point(imageData, normalise=normalise)


def dist_to_y_2d_centre_spatial_info(imageData, normalise=False):
    """returns array of 2D distances to centre and the y distance"""
    shape = numpy.asarray(imageData.shape)
    # use center point, need to do -1 because shape coordinates counts from 0, but shape counts from 1
    referencePoint = (shape - 1) / 2
    referencePoint[1] = 0
    coordinates = numpy.asarray(get_coordinates(imageData, originPoint=referencePoint), numpy.float)
    distances = numpy.asarray([numpy.zeros(shape, numpy.float), coordinates[1]])
    for x in xrange(shape[0]):
        for y in xrange(shape[1]):
            for z in xrange(shape[2]):
                distances[0, x, y, z] = numpy.linalg.norm([coordinates[0, x, y, z], coordinates[2, x, y, z]])

    if normalise:
        distances[0] /= numpy.max(distances[0])
        distances[1] /= (shape[1] - 1)
    return distances