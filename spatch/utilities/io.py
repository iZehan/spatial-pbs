"""
Created on 7 Jan 2014

@author: zw606
"""
from os.path import basename

__author__ = 'zw606'

import numpy
import nibabel.nifti1 as nib
import glob
import os
import cPickle as pickle


def construct_datafiles_list(dataPath, labelsPath, nameContains="*.nii.gz"):
    """
        Return list of files in (data, label) pairing
        - the file names are the same but exist in seperate folders
    """
    dataset = []
    dataFiles = glob.glob(dataPath + nameContains)
    labelsFiles = glob.glob(labelsPath + nameContains)
    for x in dataFiles:
        baseName = os.path.basename(x)
        if labelsPath + baseName in labelsFiles:
            dataset.append((x, labelsPath + baseName))
    return dataset


def auto_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_sub_name(filePath):
    return basename(filePath).split('.')[0]


def get_header(filePath):
    image = nib.load(filePath)
    return image.get_header()


def get_data_shape(filePath):
    return nib.load(filePath).get_header().get_data_shape()[:3]


def get_affine(filePath):
    image = nib.load(filePath)
    return image.get_affine()


def get_voxel_size(filePath):
    header = nib.load(filePath).get_header()
    return abs(header.get_base_affine()[0, 0]), abs(header.get_base_affine()[1, 1]), abs(header.get_base_affine()[2, 2])


def open_image(filePath):
    try:
        image = nib.load(filePath)
        imageData = image.get_data()
    except:
        print "Failed to open image:", filePath
        raise
    imageData = numpy.squeeze(imageData)
    return imageData


def save_3d_data(data, affine, filePath):
    try:
        img = nib.Nifti1Image(numpy.expand_dims(data, axis=3), affine)
        img.to_filename(filePath)
    except:
        print "Failed to save data to", filePath
        print "data shape:", data.shape
        raise
    return


def save_3d_labels_data(data, affine, filePath):
    try:
        img = nib.Nifti1Image(numpy.expand_dims(numpy.uint8(data), axis=3), affine)
        img.to_filename(filePath)
    except:
        print "Failed to save data to", filePath
        print "data shape:", data.shape
        raise
    return


def save_multi_folder(dataset, affine, folders, filename):
    for i in xrange(len(dataset)):
        save_3d_data(dataset[i], affine, folders[i] + filename)


def save_tree(tree, path):
    pickle.dump(tree, open(path, "w"), protocol=2)


def flip_x_save(data, filePath):
    img = nib.Nifti1Image(numpy.expand_dims(data, axis=3), numpy.eye(4))
    img.to_filename(filePath)
    return


def get_file_names(folder, nameContains="*.nii.gz"):
    dataFiles = glob.glob(folder + nameContains)
    dataFiles.sort()
    return [os.path.basename(x) for x in dataFiles]