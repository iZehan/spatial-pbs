spatial-pbs
===========

Library for patch-based segmentation (pbs) with spatial context. (Spatially Aware Patch-based Segmentation)

This library provides:

 - Functions for fast patch extraction and vectorization from images
 - An implementation of a balltree for performing multiple knn queries in parallel (using openmp)
 - Methods for providing spatial context, including using geodesic distance transforms
 - Patch-based segmentation with spatial context
 - Geodesic Patch-based Segmentation framework


Installation
============

There are some common Python dependencies - listed in requirements.txt

Simply run 'python setup.py install' to install to your python environment.


Usage
=====
Currently only supporting 3D images.

For running patch-based segmentation, due to the large size of many (3D) medical images and the large memory requirement of vectorizing patches, each image may need to be loaded individually - this is why the PatchMaker object takes in the paths to the image and labels for each atlas, patch extraction and knn search is then handled per atlas, loading each atlas one at a time (can be parallelised). Therefore having a suitable file organisation of images and atlases is required (see example in Examples/simplerun.py)

For other functions (such as spatial context, distance transforms, patch extraction), they can be used to operate on numpy arrays in memory.



Publications
============

This library (or variants) have been used for the following publications:

- Z. Wang, R. Wolz, T. Tong, D. Rueckert - "Spatially Aware Patch-Based Segmentation (SAPS): An Alternative Patch-Based Segmentation Framework" http://link.springer.com/chapter/10.1007/978-3-642-36620-8_10
- Z. Wang, C. Donoghue, D. Rueckert - "Patch-Based Segmentation without Registration: Application to Knee MRI" http://link.springer.com/chapter/10.1007/978-3-319-02267-3_13
- Z. Wang, K. Bhatia, B. Glocker, A. de Marvao, T. Dawes, K. Misawa, K. Mori, D. Rueckert - "Geodesic Patch-based Segmentation"
http://link.springer.com/chapter/10.1007/978-3-319-10404-1_83
