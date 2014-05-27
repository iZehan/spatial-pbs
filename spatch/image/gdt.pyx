#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
"""
Created on 21 Feb 2014

@author: Zehan Wang
"""
import numpy
cimport numpy
numpy.import_array()
from libc.math cimport pow, sqrt

ctypedef numpy.int16_t SHORT
ctypedef numpy.uint8_t BOOL


__all__= ["geodesic_distance_transform"]

def geodesic_distance_transform(binaryImage, imageMagnitudes, numIterations=3, spacing=(1.0, 1.0, 1.0),
                                includeEDT=True):
    """
    Calculates geodesic distance transform given an binary mask & image gradient magnitudes
    by default (includeEDT), includes euclidean distance + geodesic gradient/magnitude distance
    """
    cdef double[:, :, ::1] sqDistLookUp
    cdef int i, j, k

    cdef numpy.ndarray[double, ndim=3] distanceMap = numpy.zeros(binaryImage.shape, dtype=numpy.float64, order='c')
    distanceMap[binaryImage==0] = numpy.inf
    cdef double[:, :, :] distanceMapView = distanceMap

    cdef double[:, :, :] imageMagnitudesView = numpy.asarray(imageMagnitudes, dtype=numpy.float64, order='c')

    if includeEDT:
        imageMagnitudesView = numpy.power(imageMagnitudesView, 2)
        sqDistLookUp = numpy.zeros((3, 3, 3), dtype=numpy.float64, order='c')
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    sqDistLookUp[i, j, k] = ((i-1)*spacing[0])**2 + ((j-1)*spacing[1])**2 + ((k-1)*spacing[2])**2

        for _ in range(numIterations):
            gdt_pass(distanceMap, imageMagnitudesView, sqDistLookUp)
            gdt_pass(distanceMap[::-1, ::-1, ::-1], imageMagnitudesView[::-1, ::-1, ::-1], sqDistLookUp)
    else:
        for _ in range(numIterations):
            gdt_pass_no_edt(distanceMap, imageMagnitudesView)
            gdt_pass_no_edt(distanceMap[::-1, ::-1, ::-1], imageMagnitudesView[::-1, ::-1, ::-1])

    return distanceMap


cdef void gdt_pass(double[:, :, :] distanceMap, double[:, :, :]sqImageMagnitudes,
                   double[:, :, ::1] sqDistanceLookUp) nogil:
    cdef Py_ssize_t i, j, k, dx, dy, tempI, tempJ, tempK, shape0, shape1, shape2
    shape0, shape1, shape2 = distanceMap.shape[0], distanceMap.shape[1], distanceMap.shape[2]
    cdef double minDistance, distance
    for i in range(shape0):
        for j in range(shape1):
            for k in range(shape2):
                minDistance = distanceMap[i,j,k]
                if i > 0:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            tempI, tempJ, tempK = i-1, j+dy, k+dx
                            if 0 <= tempJ < shape1 and 0 <= tempK < shape2:
                                distance = distanceMap[tempI, tempJ, tempK] \
                                           + sqrt(sqDistanceLookUp[0, dy + 1, dx + 1]
                                                 + sqImageMagnitudes[tempI, tempJ, tempK])
                                if distance < minDistance:
                                    minDistance = distance
                if j > 0:
                    for dx in range(-1, 2):
                        tempI, tempJ, tempK = i, j-1, k+dx
                        if 0 <= tempK < shape2:
                            distance = distanceMap[tempI, tempJ, tempK] \
                                       + sqrt(sqDistanceLookUp[1, 0, dx + 1] + sqImageMagnitudes[tempI, tempJ, tempK])
                            if distance < minDistance:
                                minDistance = distance
                if k > 0:
                    tempI, tempJ, tempK = i, j, k-1
                    distance = distanceMap[tempI, tempJ, tempK] \
                               + sqrt(sqDistanceLookUp[1, 1, 0] + sqImageMagnitudes[tempI, tempJ, tempK])
                    if distance < minDistance:
                        minDistance = distance

                distanceMap[i, j, k] = minDistance


cdef void gdt_pass_no_edt(double[:, :, :] distanceMap, double[:, :, :]imageMagnitudes) nogil:
    cdef Py_ssize_t i, j, k, dx, dy, tempI, tempJ, tempK, shape0, shape1, shape2
    shape0, shape1, shape2 = distanceMap.shape[0], distanceMap.shape[1], distanceMap.shape[2]
    cdef double minDistance, distance
    for i in range(shape0):
        for j in range(shape1):
            for k in range(shape2):
                minDistance = distanceMap[i,j,k]
                if i > 0:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            tempI, tempJ, tempK = i-1, j+dy, k+dx
                            if 0 <= tempJ < shape1 and 0 <= tempK < shape2:
                                distance = distanceMap[tempI, tempJ, tempK]
                                distance += imageMagnitudes[tempI, tempJ, tempK]
                                if distance < minDistance:
                                    minDistance = distance
                if j > 0:
                    for dx in range(-1, 2):
                        tempI, tempJ, tempK = i, j-1, k+dx
                        if 0 <= tempK < shape2:
                            distance = distanceMap[tempI, tempJ, tempK]
                            distance += imageMagnitudes[tempI, tempJ, tempK]
                            if distance < minDistance:
                                minDistance = distance

                if k > 0:
                    tempI, tempJ, tempK = i, j, k-1
                    distance = distanceMap[tempI, tempJ, tempK]
                    distance += imageMagnitudes[tempI, tempJ, tempK]
                    if distance < minDistance:
                        minDistance = distance

                distanceMap[i, j, k] = minDistance
