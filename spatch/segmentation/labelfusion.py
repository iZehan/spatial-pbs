"""
Created on 3 Dec 2012

@author: Zehan Wang
"""
from __future__ import division
import numpy

# added constant to stop divide by 0 error
from spatch.image.intensity import rescale_data

__all__ = ["non_local_means_presq", "non_local_means"]

EPSILON = 0.0001


def get_weight(distances, hSquared):
    """
        weighting =  exp((sum(||x-y||)^2)/h^2)
    """
    distances *= distances
    distances /= hSquared
    return numpy.exp(-distances)


def get_weight_presq(distances, hSquared):
    """
        weighting =  exp((sum(||x-y||)^2)/h^2)
    """
    distances /= hSquared
    return numpy.exp(-distances)


def get_h(distances, epsilon=EPSILON):
    h = numpy.min(distances)
    return (h * h) + epsilon


def get_h_presq(distances, epsilon=EPSILON):
    return numpy.min(distances) + epsilon


def get_h_presq_many(distances, epsilon=EPSILON):
    return distances[..., 0].min(0) + epsilon


def non_local_means(labels, distances, k):
    """
        weighting =  exp((sum(||x-y||)^2)/h^2)
    """
    try:
        distances = numpy.asarray(distances)
        distances.sort()
    except ValueError:
        distances = dict((labels[i], distances[i]) for i in range(len(distances)) if len(distances[i]) >= k)
        labels = distances.keys()
        distances = numpy.asarray(distances.values())
        distances.sort()
        # get h value
    try:
        hVal = get_h(distances[:, 0])
    except:
        print distances
        print labels
        raise
    weights = get_weight(distances[:, :k], hVal)
    weights = numpy.sum(weights, 1)
    try:
        return labels[numpy.argmax(weights)]
    except TypeError:
        return numpy.argmax(weights)
    except IndexError:
        print labels
        print weights
        print distances
        raise


def non_local_means_presq(labels, distances, k):
    """
        use when distances are already squared
        weighting =  exp(-(sum(||x-y||)^2)/h^2)
    @param labels: list of labels
    @param distances: list of distances for each label
    @param k: number of distances to perform local means label fusion on
    """
    try:
        distances = numpy.asarray(distances, dtype=numpy.float64)
        distances.sort()
        distances = distances[:, :k]
    except ValueError:
        distances = dict((labels[i], numpy.sort(distances[i])[..., :k]) for i in range(len(distances))
                         if len(distances[i]) >= k)
        labels = distances.keys()
        distances = numpy.asarray(distances.values())
        # get h value
    try:
        hVal = get_h_presq(distances[:, 0])
    except:
        print distances
        print labels
        raise
    weights = get_weight_presq(distances, hVal)
    weights = numpy.sum(weights, 1)
    try:
        return labels[numpy.argmax(weights)]
    except TypeError:
        return numpy.argmax(weights)
    except IndexError:
        print labels
        print weights
        print distances
        raise


def non_local_means_presq_many(labels, distances, k):
    """
    distances is a LxNxk array for L labels, N rows, k distances
    """
    try:
        distances = numpy.asarray(distances, dtype=numpy.float64)
        distances.sort()
        distances = distances[:, :k]
    except ValueError:
        distances = dict((labels[i], numpy.sort(distances[i])[..., :k]) for i in xrange(len(distances))
                         if len(distances[i]) >= k)
        labels = distances.keys()
        distances = numpy.asarray(distances.values())

    # get h value
    hValues = get_h_presq_many(distances)
    weights = [get_weight_presq(distances[:, i], hValues[i]) for i in xrange(len(distances))]
    weights = numpy.sum(weights, 2)
    maxWeights = numpy.argmax(weights, 1)
    try:
        return [labels[argMaxWeight] for argMaxWeight in maxWeights]
    except TypeError:
        return maxWeights
    except IndexError:
        print labels
        print weights
        print distances
        raise


def _combine_weights(distances, hValues, k):
    distances = numpy.exp(-distances/hValues)
    distances = numpy.prod(distances, 0)
    distances.sort()
    return distances[-k:]


def separate_gaussians_presq(labels, distances, k, epsilon=EPSILON):
    """
        use when distances are already squared
        weighting = product(exp(-(sum(||x_i-y_i||)^2)/h_i^2))
    @param labels: list of labels
    @param distances: list of tuple of distances (1xM arrays) for each label [(distances1, distances2, ...)]
    @param k: number of distances to perform local means label fusion on
    """
    # get minimums for distances
    # divide by minimums and get product of gaussians
    # sort weights (want k biggest weights for each label)
    try:

        distances = numpy.asarray(distances, dtype=numpy.float64)
        for i in range(len(distances[0])):
            distances[:, i] = rescale_data(distances[:, i], 0, 1)
        hValues = distances.min(2).min(0) + epsilon
        # hValues = [2*numpy.var(numpy.sqrt(numpy.sort(distances[:, i])[:k])) + epsilon
        #            for i in range(len(distances[0]))]
        hValues = numpy.vstack(hValues)
        weights = numpy.exp(-distances/hValues)
        weights = numpy.prod(weights, 1)
        weights.sort()
        weights = weights[:, -k:]
    except ValueError:
        # got different number of distances for each label - need to do this for each individual item in distances list
        distances = dict((labels[i], distances[i]) for i in range(len(distances)) if len(distances[i][0]) >= k)
        labels = distances.keys()
        distances = distances.values()
        print "Num Distances:", len(distances)
        try:
            hValues = [numpy.min(distances[i], 0) for i in range(len(distances))]
        except:
            print distances[i]
            temp = numpy.asarray(distances[i])
            print temp.shape
            for j in range(len(temp)):
                print temp[j].shape
            raise
        hValues = numpy.min(hValues, 0)
        print "hValues", hValues
        hValues = numpy.vstack(numpy.min(hValues, 0)) + epsilon
        weights = [_combine_weights(distances[i], hValues, k) for i in range(len(distances))]

    weights = numpy.sum(weights, 1)
    try:
        return labels[numpy.argmax(weights)]
    except TypeError:
        return numpy.argmax(weights)
    except IndexError:
        print labels
        print weights
        print distances
        print hValues
        raise