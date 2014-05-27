"""
Created on 26/06/13

@author: Zehan Wang
contains common helper functions and classes
"""
from collections import deque
import numpy
from spatch.utilities.paralleling import AbstractConsumer


__all__ = ["DictResultsSorter"]


class DictResultsSorter(AbstractConsumer):
    def __init__(self, k, labelFuseFunction, labelsIndices, numResults):
        self.labelFuseFunction = labelFuseFunction
        self.k = k
        # maintain a dictionary of distances for each label {label : distances}
        self.distancesDict = None
        # label indices is a dictionary of {label : indices} indicating where each label is relevant for label fusion
        self.labelsIndices = labelsIndices
        self.numResults = numResults

    def process(self, incomingDict):
        if incomingDict is not None:
            if self.distancesDict is not None:
                for label in incomingDict:
                    try:
                        self.distancesDict[label] = numpy.append(self.distancesDict[label], incomingDict[label], 1)
                        self.distancesDict[label].sort()
                        self.distancesDict[label] = self.distancesDict[label][:, :self.k]
                    except KeyError:
                        self.distancesDict[label] = incomingDict[label]
            else:
                self.distancesDict = incomingDict

    def results(self):
        print "[Info]--->Processing distances and performing label fusion..."
        results = deque()
        if self.labelsIndices is not None:
            # remove any labels not present from results
            self.labelsIndices = dict((label, self.labelsIndices[label]) for label in self.labelsIndices
                                      if label in self.distancesDict)
            # setup counters for each label - indicating which element
            labelsCounter = dict((label, 0) for label in self.labelsIndices.keys())
            for i in xrange(self.numResults):
                labels = [label for label in self.labelsIndices if i in self.labelsIndices[label]]
                distances = [self.distancesDict[label][labelsCounter[label]] for label in labels]
                for label in labels:
                    # increment counters
                    labelsCounter[label] += 1
                results.append(self.labelFuseFunction(labels, distances, self.k))
        else:
            labels = self.distancesDict.keys()
            for i in xrange(self.numResults):
                try:
                    distances = [self.distancesDict[label][i] for label in labels]
                    results.append(self.labelFuseFunction(labels, distances, self.k))
                except:
                    print "Number of results expected:", self.numResults
                    print "Results received per label:"
                    for label in labels:
                        print label, ":", len(self.distancesDict[label])
                    raise
        return list(results)


def arrange_results_by_location(resultsList, locationList, dataShape, dataType=numpy.uint8):
    results = numpy.zeros(dataShape, dtype=dataType)
    if locationList is not None:
        for i in xrange(len(locationList)):
            # assign label to coordinate
            results[locationList[i][0], locationList[i][1], locationList[i][2]] = resultsList[i]
    return results


def combine_refined_to_prev_results(refinedResults, prevResults, refinementMask):
    prevResults[refinementMask] = refinedResults[refinementMask]
    return prevResults