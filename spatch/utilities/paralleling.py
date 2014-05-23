#! /usr/bin/env python
"""
Created on 27 Jan 2012

@author: Zehan Wang
"""
from __future__ import division
from abc import abstractmethod, ABCMeta
from multiprocessing.queues import SimpleQueue
import gc
import math
import multiprocessing
import numpy
import time
from utilities.misc import short_format_time


def split_into_sublist(dataList, numSplits):
    """splits the data into a sublists"""
    dataLength = int(math.floor(len(dataList) / numSplits))
    newDataList = [dataList[i * dataLength:(i + 1) * dataLength] for i in xrange(numSplits)]
    newStart = numSplits * dataLength
    remainder = len(dataList) - newStart
    for i in xrange(remainder):
        newDataList[i].append(dataList[newStart + i])
    return newDataList


def multi_process_list(data, method, numProcessors, *args):
    if len(data) < 1:
        print "[WARNING]--->Received empty list for multi-processing!"
        return None
    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    processes = [None] * numProcessors
    results = [None] * len(data)
    tempRes = SimpleQueue()
    gc.disable()
    for i in xrange(numProcessors):
        newArgs = (dataSplit[i],) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()
    lastPercent = 0
    startTime = time.time()
    dataLen = len(data)
    get = tempRes.get
    for i in xrange(len(results)):
        results[i] = get()
        percentDone = i / dataLen
        if percentDone - lastPercent >= 0.1:
            timeTaken = time.time() - startTime
            timeRemain = short_format_time((dataLen - (i + 1)) / ((i + 1) / timeTaken))
            timeTaken = short_format_time(timeTaken)
            print int(percentDone * 100), "percent done | time elapsed:", timeTaken, "| time remaining:", timeRemain
            lastPercent = percentDone
    for p in processes:
        p.join()
    if lastPercent != 100:
        print "100 percent done"
    gc.enable()
    print "Finished parallel processing list of length", len(results)
    return results


def multi_process_list_with_consumer(data, method, consumerObj, numProcessors, *args):
    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    processes = [None] * numProcessors
    results = [None] * len(data)
    tempRes = SimpleQueue()
    for i in xrange(numProcessors):
        newArgs = (dataSplit[i],) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()

    lastPercent = 0
    dataLen = len(data)
    startTime = time.time()
    get = tempRes.get
    for i in xrange(len(results)):
        consumerObj.process(get())
        percentDone = i / dataLen
        if percentDone - lastPercent >= 0.1:
            timeTaken = time.time() - startTime
            timeRemain = short_format_time((dataLen - (i + 1)) / ((i + 1) / timeTaken))
            timeTaken = short_format_time(timeTaken)
            print int(percentDone * 100), "percent done | time elapsed:", timeTaken, "| time remaining:", timeRemain
            lastPercent = percentDone
    for p in processes:
        p.join()
    if lastPercent != 100:
        print "100 percent done"
    print "Finished parallel processing list of length", len(results)
    return consumerObj.results()


class AbstractConsumer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, incoming):
        pass

    @abstractmethod
    def results(self):
        pass


def multi_process_list_no_gc(data, method, numProcessors, *args):
    if len(data) < 1:
        print "[WARNING]--->Received empty list for multi-processing!"
        return None

    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    processes = [None] * numProcessors
    results = [None] * len(data)
    tempRes = multiprocessing.Queue()
    for i in xrange(numProcessors):
        newArgs = (dataSplit[i],) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()
    lastPercent = 0
    for i in range(len(results)):
        results[i] = (tempRes.get())
        percentDone = int(i / len(data) * 100)
        if percentDone - lastPercent >= 10:
            print percentDone, "percent done"
            lastPercent = percentDone
    for p in processes:
        p.join()
    if lastPercent != 100:
        print "100 percent done"
    print "Finished parallel processing list of length", len(results)
    return results


def multi_process_list_to_dict(data, method, numProcessors, *args):
    """returns a set instead of a list"""
    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    processes = [None] * numProcessors
    results = dict()
    tempRes = multiprocessing.Queue()
    gc.disable()
    for i in range(numProcessors):
        newArgs = (dataSplit[i],) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()
    lastPercent = 0
    i = 0
    while i < len(data):
        temp = tempRes.get()
        if temp in results:
            results[temp] += 1
        else:
            results[temp] = 1
        percentDone = int(i / len(data) * 100)
        if percentDone - lastPercent >= 10:
            print percentDone, "percent done"
            lastPercent = percentDone
        i += 1

    for p in processes:
        p.terminate()
    if lastPercent != 100:
        print "100 percent done"
    gc.enable()
    print "Finished parallel processing list of length", len(data)
    return results


def multi_process_list_with_ids(data, ids, method, numProcessors, *args):
    """
        data is returned from the worker/method a tuple (data, id)
        data is then put in list in order of id to be returned
    """
    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    idsSplit = split_into_sublist(ids, numProcessors)
    processes = [None] * numProcessors
    results = [None] * len(data)
    tempRes = multiprocessing.Queue()
    gc.disable()
    for i in range(numProcessors):
        newArgs = (dataSplit[i], idsSplit[i]) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()
    lastPercent = 0
    for i in xrange(len(data)):
        item, index = tempRes.get()
        results[index] = item
        percentDone = int(i / len(results) * 100)
        if percentDone - lastPercent >= 10:
            print percentDone, "percent done"
            lastPercent = percentDone
    for p in processes:
        p.join()
    if lastPercent != 100:
        print "100 percent done"
    gc.enable()
    print "Finished parallel processing list of length", len(results)
    return results


def multi_process_numpy(data, method, numProcessors, *args):
    """
        same as multi_process_list but uses a numpy array and is significantly faster
    """
    if numProcessors > len(data):
        numProcessors = len(data)
    dataSplit = split_into_sublist(data, numProcessors)
    processes = [None] * numProcessors
    results = numpy.array([None] * len(data))
    tempRes = multiprocessing.Queue()
    gc.disable()
    for i in range(numProcessors):
        newArgs = (dataSplit[i],) + args + (tempRes,)
        processes[i] = multiprocessing.Process(target=method, args=newArgs)
        processes[i].start()
    lastPercent = 0
    for i in range(len(results)):
        results[i] = (tempRes.get())
        percentDone = int(i / len(results) * 100)
        if percentDone - lastPercent >= 10:
            print percentDone, "percent done"
            lastPercent = percentDone
    for p in processes:
        p.terminate()
    if lastPercent != 100:
        print "100 percent done"
    gc.enable()
    print "Finished parallel processing list of length", len(results)
    return results