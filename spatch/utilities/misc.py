import collections
import time
import numpy


__all__ = ["glob_escape", "auto_specify_labels", "auto_non_background_labels", "extend_dict_values"]


def glob_escape(pathString):
    """
    if [] in path, need to change so glob.glob will not read it as regex,
    """
    return pathString.replace('[', '[[]')


def short_format_time(t):
    if t > 3600:
        return "%4.1f hours" % round(t / 3600., 1)
    if t > 60:
        return "%2.1f mins" % round(t / 60., 1)
    else:
        return "%2.1f secs" % round(t, 1)


def time_it(someFunction):
    def inner(*args, **kwargs):
        startTime = time.time()
        someFunction(*args, **kwargs)
        timeTaken = time.time() - startTime
        print "Time taken:", round(timeTaken, 2), "seconds"

    return inner


def auto_specify_labels(labelledData):
    return numpy.unique(labelledData)


def auto_non_background_labels(labelledData):
    labels = numpy.unique(labelledData)
    return labels[labels > 0]


def extend_dict_values(dict1, dict2):
    """
    adds extends dict1 with values from dict2 - creates list if multiple values or
    """
    for k in dict2:
        if isinstance(dict2[k], collections.Iterable):
            try:
                dict1[k].extend(dict2[k])
            except KeyError:
                dict1[k] = dict2[k]
            except AttributeError:
                dict1[k] = [dict1[k]].extend(dict2[k])
        else:
            try:
                dict1[k].append(dict2[k])
            except KeyError:
                dict1[k] = dict2[k]
            except AttributeError:
                dict1[k] = [dict1[k], dict2[k]]
    return dict1