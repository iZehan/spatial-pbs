"""
Created on 05/12/13

@author: zw606
"""
import glob
from os.path import join, basename
from spatch.image import spatialcontext
from spatch.image.mask import get_boundary_mask
from spatch.segmentation.patchbased import SAPS
from spatch.utilities.io import open_image, get_affine, save_3d_labels_data
from spatch.image.spatialcontext import NORMALISED_COORDINATES, GDT

INITIAL_SPATIAL_INFO = NORMALISED_COORDINATES
REFINEMENT_SPATIAL_INFO = GDT
IMAGE_EXPAND = True


def get_subject_id(fileName):
    nameParts = fileName.split('.')[0].split('_')
    return nameParts[0]


def initial_saps_segment(trainingSet, targetFile, imagesPath, labelsPath, patchSize, k, spatialWeight,
                         spatialInfoType=INITIAL_SPATIAL_INFO, maskData=None, numProcessors=21):
    targetImage = open_image(join(imagesPath, targetFile))

    print "ImageShape:", targetImage.shape
    # Ensure target subject is not included in atlases
    targetId = get_subject_id(targetFile)
    trainingSet = [x for x in trainingSet if get_subject_id(x) != targetId]

    saps = SAPS(imagesPath, labelsPath, patchSize, boundaryDilation=None,
                spatialWeight=spatialWeight, minValue=None, maxValue=None,
                spatialInfoType=spatialInfoType)

    # get results
    results = saps.label_image(targetImage, k, trainingSet, queryMaskDict=maskData, numProcessors=numProcessors)

    return results


def refinement_saps_segment(trainingSet, targetFile, imagesPath, labelsPath, patchSize, k, spatialWeight,
                            prevResultsPath, dtLabels, boundaryRefinementSize=2, dtErosion=None,
                            spatialInfoType=REFINEMENT_SPATIAL_INFO, numProcessors=21):
    targetImage = open_image(join(imagesPath, targetFile))

    print "ImageShape:", targetImage.shape
    # Ensure target subject is not included in atlases
    targetId = get_subject_id(targetFile)
    trainingSet = [x for x in trainingSet if get_subject_id(x) != targetId]

    saps = SAPS(imagesPath, labelsPath, patchSize, boundaryDilation=boundaryRefinementSize,
                spatialWeight=spatialWeight, minValue=None, maxValue=None,
                spatialInfoType=spatialInfoType)

    prevResults = open_image(join(prevResultsPath, targetFile))
    refinementMask = get_boundary_mask(prevResults, boundaryRefinementSize)
    queryMaskDict = {1: refinementMask}

    if dtErosion is None:
        dtErosion = boundaryRefinementSize

    spatialInfo = spatialcontext.get_dt_spatial_context_dict(prevResults, spatialInfoType=spatialInfoType,
                                                             spatialLabels=dtLabels, labelErosion=dtErosion,
                                                             imageData=targetImage).values()

    # get results
    results = saps.label_image(targetImage, k, trainingSet, queryMaskDict=queryMaskDict, spatialInfo=spatialInfo,
                               dtLabels=dtLabels, numProcessors=numProcessors)

    return results


def run_leave_one_out(imagesPath, labelsPath, savePath, patchSize=7, k=15, spatialWeight=400,
                      prevResultsPath=None, dtLabels=None, refinementSize=2,
                      numProcessors=8, fileName="*.nii.gz"):
    files = glob.glob(join(imagesPath, fileName))
    print "Number of files found:", len(files)
    dataset = [basename(x) for x in files]

    if prevResultsPath is not None:
        for targetFile in dataset:
            trainingSet = [x for x in dataset if x != targetFile]
            results = refinement_saps_segment(trainingSet, targetFile, imagesPath, labelsPath,
                                              patchSize, k, spatialWeight,
                                              prevResultsPath, dtLabels, boundaryRefinementSize=refinementSize,
                                              numProcessors=numProcessors)
            save_3d_labels_data(results, get_affine(join(imagesPath, targetFile)),
                                join(savePath, targetFile))
    else:
        for targetFile in dataset:
            trainingSet = [x for x in dataset if x != targetFile]
            results = initial_saps_segment(trainingSet, targetFile, imagesPath, labelsPath,
                                           patchSize, k, spatialWeight, numProcessors=numProcessors)
            save_3d_labels_data(results, get_affine(join(imagesPath, targetFile)),
                                join(savePath, targetFile))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--imagesPath", default=None,
                        help="Set images to use")
    parser.add_argument("--labelsPath", default=None,
                        help="Set labels to use")
    parser.add_argument("--savePath", default=None,
                        help="Set path to initial results")
    parser.add_argument("--prevResultsPath", default=None,
                        help="Set path to results folder that can be used")

    parser.add_argument("--fileName", default="*.nii.gz",
                        help="Specify which files to work on (takes regex)")

    parser.add_argument("--patchSize", type=int, default=7, nargs="+",
                        help="Set the patch size to use")
    parser.add_argument("-k", type=int, default=15,
                        help="Set number of nearest neighbours to use")
    parser.add_argument("--spatialWeight", type=float, default=10,
                        help="Set path to initial results")
    parser.add_argument("--dtLabels", type=int, default=None, nargs="+",
                        help="Set the labels (structures) to use to provide adaptive spatial context")
    parser.add_argument("--refinementSize", type=int, default=2,
                        help="Set boundary size for refinement (number of dilations-erosions used)")

    parser.add_argument("--numProcessors", type=int, default=10,
                        help="Set number of processors to use")

    options = parser.parse_args()

    run_leave_one_out(options.imagesPath, options.labelsPath, options.savePath, patchSize=options.patchSize,
                      k=options.k, prevResultsPath=options.prevResultsPath, dtLabels=options.dtLabels,
                      spatialWeight=options.spatialWeight, numProcessors=options.numProcessors,
                      fileName=options.fileName, refinementSize=options.refinementSize)
