#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
Created on 19 Dec 2012

@author: Zehan Wang
@Notes: modified from scikit-learn BallTree
=========
Ball Tree
=========
A ball tree is a data object which speeds up nearest neighbor
searches in high dimensions (see scikit-learn neighbors module
documentation for an overview of neighbor trees). There are many
types of ball trees.  This package provides a basic implementation
in cython.

Implementation Notes
--------------------

A ball tree can be thought of as a collection of nodes.  Each node
stores a centroid, a radius, and the pointers to two child nodes.

* centroid : the centroid of a node is the mean of all the locations
    of points within the node
* radius : the radius of a node is the distance from the centroid
    to the furthest point in the node.
* subnodes : each node has a maximum of 2 child nodes.  The data within
    the parent node is divided between the two child nodes.

In a typical tree implementation, nodes may be classes or structures which
are dynamically allocated as needed.  This offers flexibility in the number
of nodes, and leads to very straightforward and readable code.  It also means
that the tree can be dynamically augmented or pruned with new data, in an
in-line fashion.  This approach generally leads to recursive code: upon
construction, the head node constructs its child nodes, the child nodes
construct their child nodes, and so-on.

The current package uses a different approach: all node data is stored in
a set of numpy arrays which are pre-allocated.  The main advantage of this
approach is that the whole object can be quickly and easily saved to disk
and reconstructed from disk.  This also allows for an iterative interface
which gives more control over the heap, and leads to speed.  There are a
few disadvantages, however: once the tree is built, augmenting or pruning it
is not as straightforward.  Also, the size of the tree must be known from the
start, so there is not as much flexibility in building it.

BallTree Pseudo-code
~~~~~~~~~~~~~~~~~~~~
Because understanding a ball tree is simpler with recursive code, here is some
pseudo-code to show the structure of the main functionality

    # Ball Tree pseudo code

    class Node:
        #class data:
        centroid
        radius
        child1, child2

        #class methods:
        def construct(data):
            centroid = compute_centroid(data)
            radius = compute_radius(centroid, data)

            # Divide the data into two approximately equal sets.
            # This is often done by splitting along a single dimension.
            data1, data2 = divide(data)

            if number_of_points(data1) > 0:
                child1.construct(data1)

            if number_of_points(data2) > 0:
                child2.construct(data2)

        def query(pt, neighbors_heap):
            # compute the minimum distance from pt to any point in this node
            d = distance(point, centroid)
            if d < radius:
                min_distance = 0
            else:
                min_distance = d - radius

            if min_distance > max_distance_in(neighbors_heap):
                # all these points are too far away.  cut off the search here
                return
            elif node_size > 1:
                child1.query(pt, neighbors_heap)
                child2.query(pt, neighbors_heap)


    object BallTree:
        #class data:
        data
        root_node

        #class methods
        def construct(data, num_leaves):
            root_node.construct(data)

        def query(point, num_neighbors):
            neighbors_heap = empty_heap_of_size(num_neighbors)
            root_node.query(point, neighbors_heap)

This certainly is not a complete description, but should give the basic idea
of the form of the algorithm.  The implementation below is much faster than
anything mirroring the pseudo-code above, but for that reason is much more
opaque.  Here's the basic idea:

BallTree Storage
~~~~~~~~~~~~~~~~
The BallTree information is stored using a combination of
"Array of Structures" and "Structure of Arrays" to maximize speed.
Given input data of size ``(n_samples, n_features)``, BallTree computes the
expected number of nodes ``n_nodes`` (see below), and allocates the
following arrays:

* ``data`` : a float array of shape ``(n_samples, n_features)``
    This is simply the input data.  If the input matrix is well-formed
    (contiguous, c-ordered, correct data type) then no copy is needed
* ``idx_array`` : an integer array of size ``n_samples``
    This can be thought of as an array of pointers to the data in ``data``.
    Rather than shuffling around the data itself, we shuffle around pointers
    to the rows in data.
* ``node_centroid_arr`` : a float array of shape ``(n_nodes, n_features)``
    This stores the centroid of the data in each node.
* ``node_info_arr`` : a size-``n_nodes`` array of ``NodeInfo`` structures.
    This stores information associated with each node.  Each ``NodeInfo``
    instance has the following attributes:
    - ``idx_start``
    - ``idx_end`` : ``idx_start`` and ``idx_end`` reference the part of
      ``idx_array`` which point to the data associated with the node.
      The data in node with index ``i_node`` is given by
      ``data[idx_array[idx_start:idx_end]]``
    - ``is_leaf`` : a boolean value which tells whether this node is a leaf:
      that is, whether or not it has children.
    - ``radius`` : a floating-point value which gives the distance from
      the node centroid to the furthest point in the node.

One feature here is that there are no stored pointers from parent nodes to
child nodes and vice-versa.  These pointers are implemented implicitly:
For a node with index ``i``, the two children are found at indices
``2 * i + 1`` and ``2 * i + 2``, while the parent is found at index
``floor((i - 1) / 2)``.  The root node has no parent.

With this data structure in place, the functionality of the above BallTree
pseudo-code can be implemented in a much more efficient manner.
Most of the data passing done in this code uses raw data pointers.
Using numpy arrays would be preferable for safety, but the
overhead of array slicing and sub-array construction leads to execution
time which is several orders of magnitude slower than the current
implementation.

Priority Queue vs Max-heap
~~~~~~~~~~~~~~~~~~~~~~~~~~
When querying for more than one neighbor, the code must maintain a list of
the current k nearest points.  The BallTree code implements this in two ways.

- A priority queue: this is just a sorted list.  When an item is added,
  it is inserted in the appropriate location.  The cost of the search plus
  insert averages O[k].
- A max-heap: this is a binary tree structure arranged such that each node is
  greater than its children.  The cost of adding an item is O[log(k)].
  At the end of the iterations, the results must be sorted: a quicksort is
  used, which averages O[k log(k)].  Quicksort has worst-case O[k^2]
  performance, but because the input is already structured in a max-heap,
  the worst case will not be realized.  Thus the sort is a one-time operation
  with cost O[k log(k)].

Each insert is performed an average of log(N) times per query, where N is
the number of training points.  Because of this, for a single query, the
priority-queue approach costs O[k log(N)], and the max-heap approach costs
O[log(k)log(N)] + O[k log(k)].  Tests show that for sufficiently large k,
the max-heap approach out-performs the priority queue approach by a factor
of a few.  In light of these tests, the code uses a priority queue for
k < 5, and a max-heap otherwise.

Memory Allocation
~~~~~~~~~~~~~~~~~
It is desirable to construct a tree in as balanced a way as possible.
Given a training set with n_samples and a user-supplied leaf_size, if
the points in each node are divided as evenly as possible between the
two children, the maximum depth needed so that leaf nodes satisfy
``leaf_size <= n_points <= 2 * leaf_size`` is given by
``n_levels = 1 + max(0, floor(log2((n_samples - 1) / leaf_size)))``
(with the exception of the special case where ``n_samples < leaf_size``)
For a given number of levels, the number of points in a tree is given by
``n_nodes = 2 ** n_levels - 1``.  Both of these results can be shown
by induction.  Using them, the correct amount of memory can be pre-allocated
for a given ``n_samples`` and ``leaf_size``.
"""
import numpy as np
cimport numpy as np
np.import_array()

from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport fabs, pow, sqrt


__all__ = ["BallTree"]

######################################################################
# global definitions
#
# type used for data
NUMPY_DATA_TYPE = np.float64
ctypedef double DATA_TYPE_C

# add to difference if comparing with NaN
cdef double NAN_DIFF = 10

# type used for indices & counts
# warning: there will be problems if this is switched to an unsigned type!
NUMPY_INDEX_TYPE = np.int32
ctypedef int INDEX_TYPE_C

# infinity
cdef DATA_TYPE_C infinity = np.inf


######################################################################
# utility functions: fast max, min, and absolute value
######################################################################
cdef inline DATA_TYPE_C dmax(DATA_TYPE_C x, DATA_TYPE_C y) nogil:
    return max(x, y)


cdef inline DATA_TYPE_C dmin(DATA_TYPE_C x, DATA_TYPE_C y) nogil:
    return min(x, y)


######################################################################
# distance functions
#  These implement the Minkowski p-distance given by
#    dist = sum((x - y) ** p) ** (1 / p)
#  To compare distances, the raising to the (1 / p) is not necessary
#  therefore, for speed, we also define a function dist_p() given by
#    dist_p = sum((x - y) ** p)
#  there are also functions dist_from_dist_p() and dist_p_from_dist()
#  which convert between these forms.
# UPDATE: no longer using dist -> remove complexity all dists will be given as dist_p
######################################################################


cdef inline DATA_TYPE_C squared_euclidean_distance(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                                   INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r, d
    r = 0
    for i in range(n):
        d = x1[i] - x2[i]
        r += d * d
    return r

cdef inline DATA_TYPE_C p_powered_minkowski_distance(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                                     INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r, d
    r = 0
    for i in range(n):
        d = fabs(x1[i] - x2[i])
        r += pow(d, p)
    return r

cdef inline DATA_TYPE_C minkowski_distance(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                           INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r, d
    r = 0
    for i in range(n):
        d = fabs(x1[i] - x2[i])
        r += pow(d, p)
    return pow(r, 1./p)

cdef inline DATA_TYPE_C manhattan_distance(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                           INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r
    r = 0
    for i in range(n):
        r += fabs(x1[i] - x2[i])
    return r

cdef inline DATA_TYPE_C euclidean_distance(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                           INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r, d
    r = 0
    for i in range(n):
        d = x1[i] - x2[i]
        r += d * d
    return sqrt(r)

cdef inline DATA_TYPE_C dist_p_inf(DATA_TYPE_C *x1, DATA_TYPE_C *x2,
                                   INDEX_TYPE_C n, DATA_TYPE_C p) nogil:
    cdef INDEX_TYPE_C i
    cdef DATA_TYPE_C r
    r = 0
    for i in range(n):
        r = dmax(r, fabs(x1[i] - x2[i]))
    return r

cdef inline DATA_TYPE_C dist_p_from_dist(DATA_TYPE_C r,
                                         DATA_TYPE_C p) nogil:
    if p == 2:
        return r * r
    elif p == infinity:
        return r
    elif p == 1:
        return r
    else:
        return pow(r, p)

######################################################################
# NodeInfo struct
#  used to keep track of node information.
#  there is also a centroid for each node: this is kept in a separate
#  array for efficiency.  This is a hybrid of the "Array of Structures"
#  and "Structure of Arrays" styles.
cdef struct NodeInfo:
    INDEX_TYPE_C idx_start
    INDEX_TYPE_C idx_end
    INDEX_TYPE_C is_leaf
    DATA_TYPE_C radius


######################################################################
# stack struct
#  This is used to keep track of the recursion stack in Node_query
cdef struct stack_item:
    INDEX_TYPE_C i_node
    DATA_TYPE_C dist_p_LB


cdef struct stack:
    int n
    stack_item* heap
    int size

cdef inline void stack_create(stack* self, int size) nogil:
    self.size = size
    self.heap = <stack_item*> malloc(sizeof(stack_item) * size)
    self.n = 0

cdef inline void stack_destroy(stack* self) nogil:
    free(self.heap)


cdef inline void stack_resize(stack* self, int new_size) nogil:
    self.size = new_size
    self.heap = <stack_item*>realloc(<void*> self.heap,
                                            new_size * sizeof(stack_item))

cdef inline void stack_push(stack* self, stack_item item) nogil:
    if self.n >= self.size:
        stack_resize(self, 2 * self.size + 1)

    self.heap[self.n] = item
    self.n += 1

cdef inline stack_item stack_pop(stack* self) nogil:
    self.n -= 1
    return self.heap[self.n]


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)

######################################################################
# BallTree class
#
cdef class BallTree(object):
    """
    Ball Tree for fast nearest-neighbor searches :

    BallTree(X, leaf_size=20, p=2.0)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        n_samples is the number of points in the data set, and
        n_features is the dimension of the parameter space.
        Note: if X is a C-contiguous array of doubles then data will
        not be copied. Otherwise, an internal copy will be made.

    leaf_size : positive integer (default = 20)
        Number of points at which to switch to brute-force. Changing
        leaf_size will not affect the results of a query, but can
        significantly impact the speed of a query and the memory required
        to store the built ball tree.  The amount of memory needed to
        store the tree scales as
        2 ** (1 + floor(log2((n_samples - 1) / leaf_size))) - 1
        For a specified ``leaf_size``, a leaf node is guaranteed to
        satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
        the case that ``n_samples < leaf_size``.

    p : distance metric for the BallTree.  ``p`` encodes the Minkowski
        p-distance::

            D = sum((X[i] - X[j]) ** p) ** (1. / p)

        p must be greater than or equal to 1, so that the triangle
        inequality will hold.  If ``p == np.inf``, then the distance is
        equivalent to::

            D = max(X[i] - X[j])

    Attributes
    ----------
    data : np.ndarray
        The training data

    warning_flag : bool (removed - not needed)
        Warning flag is set to true during query(...) if results are
        dependent on the order of the training cases.
        For classification or regression based on k-neighbors, if
        neighbor k and neighbor k+1 have identical distances but different
        labels, then the result will be dependent on the ordering of the
        training data.  In this case, ``warning_flag`` will be set to True.

    Examples
    --------
    Query for k-nearest neighbors

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        >>> ball_tree = BallTree(X, leaf_size=2)              # doctest: +SKIP
        >>> dist, ind = ball_tree.query(X[0], n_neighbors=3)  # doctest: +SKIP
        >>> print ind  # indices of 3 closest neighbors
        [0 3 1]
        >>> print dist  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]

    Pickle and Unpickle a ball tree (using protocol = 2).  Note that the
    state of the tree is saved in the pickle operation: the tree is not
    rebuilt on un-pickling

        >>> import numpy as np
        >>> import pickle
        >>> np.random.seed(0)
        >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        >>> ball_tree = BallTree(X, leaf_size=2)          # doctest: +SKIP
        >>> s = pickle.dumps(ball_tree, protocol=2)       # doctest: +SKIP
        >>> ball_tree_copy = pickle.loads(s)              # doctest: +SKIP
        >>> dist, ind = ball_tree_copy.query(X[0], k=3)   # doctest: +SKIP
        >>> print ind  # indices of 3 closest neighbors
        [0 3 1]
        >>> print dist  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]
    """
    cdef readonly np.ndarray data
    cdef np.ndarray idx_array
    cdef np.ndarray node_centroid_arr
    cdef np.ndarray node_info_arr
    cdef DATA_TYPE_C p
    # True Distance function - required for node radius comparisons due to triangle inequality for distance metrics
    cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
    # Pseudo distance function - used for leaf items to order distances by, cheaper to execute (no rooting)
    cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
    cdef INDEX_TYPE_C leaf_size
    cdef INDEX_TYPE_C n_levels
    cdef INDEX_TYPE_C n_nodes
    cdef INDEX_TYPE_C numDimensions
    cdef INDEX_TYPE_C numItems

    def __cinit__(self):
        """
        initialize all arrays to empty.  This will prevent memory errors
        in rare cases where __init__ is not called
        """
        self.data = np.empty((0,0), dtype=NUMPY_DATA_TYPE)
        self.idx_array = np.empty(0, dtype=NUMPY_INDEX_TYPE)
        self.node_centroid_arr = np.empty((0,0), dtype=NUMPY_DATA_TYPE)
        self.node_info_arr = np.empty(0, dtype='c')

    def __init__(self, X, INDEX_TYPE_C leaf_size=20, DATA_TYPE_C p=2):
        self.data = np.asarray(X, dtype=NUMPY_DATA_TYPE, order='C')

        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if self.data.ndim != 2:
            raise ValueError("X should have two dimensions")

        if p < 0:
            raise ValueError("p must be greater than 0")
        self.p = p
        if p == 1:
            self.distance_function = &manhattan_distance
            self.pseudo_distance_function = &manhattan_distance
        elif p == 2:
            self.distance_function = &euclidean_distance
            self.pseudo_distance_function = &squared_euclidean_distance
        elif p == infinity:
            self.distance_function = &dist_p_inf
            self.pseudo_distance_function = &dist_p_inf
        else:
            self.distance_function = &minkowski_distance
            self.pseudo_distance_function = &p_powered_minkowski_distance

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size

        self.numItems = self.data.shape[0]
        self.numDimensions = self.data.shape[1]

        # determine number of levels in the ball tree, and from this
        # the number of nodes in the ball tree
        self.n_levels = np.log2(max(1, (self.numItems - 1)/self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

        self.idx_array = np.arange(self.numItems, dtype=NUMPY_INDEX_TYPE)

        self.node_centroid_arr = np.empty((self.n_nodes, self.numDimensions), dtype=NUMPY_DATA_TYPE, order='C')

        self.node_info_arr = np.empty(self.n_nodes * sizeof(NodeInfo), dtype='c', order='C')
        self.build_tree_()

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return newObj, (BallTree,), self.__getstate__()

    def __getstate__(self):
        """
        get state for pickling
        """
        return (self.data,
                self.idx_array,
                self.node_centroid_arr,
                self.node_info_arr,
                self.p,
                self.leaf_size,
                self.n_levels,
                self.n_nodes)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.data = state[0]
        self.idx_array = state[1]
        self.node_centroid_arr = state[2]
        self.node_info_arr = state[3]
        self.p = state[4]
        self.leaf_size = state[5]
        self.n_levels = state[6]
        self.n_nodes = state[7]
        self.numItems = self.data.shape[0]
        self.numDimensions = self.data.shape[1]

    def query(self, X, INDEX_TYPE_C k=1):
        """
        query(X, k=1)

        query the Ball Tree for the k nearest neighbors

        Parameters
        ----------
        X : 1D array dimension self.dim
        k : integer  (default = 1)
            The number of nearest neighbors to return

        Returns
        -------
        (d,i)

        d : array of floats - the list of distances to the
            neighbors of the corresponding point
            (note that distances are not sorted)

        i : array of integers the list of indices of
            neighbors of the corresponding point
            (note that neighbors are not sorted)

        Examples
        --------
        Query for k-nearest neighbors

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
            >>> ball_tree = BallTree(X, leaf_size=2)    # doctest: +SKIP
            >>> dist, ind = ball_tree.query(X[0], k=3)  # doctest: +SKIP
            >>> print ind  # indices of 3 closest neighbors
            [0 3 1]
            >>> print dist  # distances to 3 closest neighbors
            [ 0.          0.19662693  0.29473397]
        """

        if len(X) != self.numDimensions:
            raise ValueError("query data dimension must match BallTree data dimension \n--> query data dimensions: "
                             + str(len(X)) + "; BallTree dimensions: " + str(self.numDimensions))

        if k > self.numItems:
            k = self.numItems
            # raise ValueError("k must be less than or equal to the number of training points")

        # for k less than 5, a priority queue is slightly faster
        # for more neighbors, a max-heap implementation is faster
        cdef INDEX_TYPE_C use_max_heap = (k >= 5)

        cdef np.ndarray[DATA_TYPE_C, ndim=1] queryData = np.asarray(X, dtype=NUMPY_DATA_TYPE, order='c')
        cdef np.ndarray[DATA_TYPE_C, ndim=1] distances = np.empty(k, dtype=NUMPY_DATA_TYPE, order='c')
        cdef np.ndarray[INDEX_TYPE_C, ndim=1] idx_array = np.empty(k, dtype=NUMPY_INDEX_TYPE, order='c')

        # initialize arrays.  This is only needed for correct behavior of the warning flag.
        idx_array[:] = -9999
        distances[:] = np.inf

        cdef DATA_TYPE_C* dist_ptr = <DATA_TYPE_C*>distances.data
        cdef INDEX_TYPE_C* idx_ptr = <INDEX_TYPE_C*>idx_array.data

        cdef stack node_stack
        stack_create(&node_stack, self.n_levels + 1)

        self.query_one_(<DATA_TYPE_C*>queryData.data, k, dist_ptr, idx_ptr, &node_stack, use_max_heap)

        # if max-heap is used, results must be sorted
        if use_max_heap:
            sort_dist_idx(dist_ptr, idx_ptr, k)

        stack_destroy(&node_stack)

        return distances, idx_array

    def query_many(self, X, INDEX_TYPE_C k=1):
        """
        query(X, k=1, return_distance=True)

        query the Ball Tree for the k nearest neighbors

        Parameters
        ----------
        @param: X - 2D array, last dimension self.dim
            An array of points to query
        @param: k - integer  (default = 1)
            The number of nearest neighbors to return
        return_distance : boolean (default = True)
            if True, return a tuple (d,i)
            if False, return array i

        Returns
        -------
        i    : if return_distance == False
        (d,i) : if return_distance == True

        d : array of doubles - shape: x.shape[:-1] + (k,)
            each entry gives the list of distances to the
            neighbors of the corresponding point
            (note that distances are not sorted)

        i : array of integers - shape: x.shape[:-1] + (k,)
            each entry gives the list of indices of
            neighbors of the corresponding point
            (note that neighbors are not sorted)

        Examples
        --------
        Query for k-nearest neighbors

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
            >>> ball_tree = BallTree(X, leaf_size=2)    # doctest: +SKIP
            >>> dist, ind = ball_tree.query(X[0], k=3)  # doctest: +SKIP
            >>> print ind  # indices of 3 closest neighbors
            [0 3 1]
            >>> print dist  # distances to 3 closest neighbors
            [ 0.          0.19662693  0.29473397]
        """

        cdef np.ndarray[DATA_TYPE_C, ndim=2] queryData = np.asarray(X, dtype=NUMPY_DATA_TYPE, order="c")

        cdef INDEX_TYPE_C numQueries = queryData.shape[0]
        cdef INDEX_TYPE_C numDimensions = queryData.shape[1]
        if numDimensions != self.numDimensions:
            raise ValueError("query data dimension must match BallTree data dimension \n--> query data dimensions: "
                             + str(numDimensions) + "; BallTree dimensions: " + str(self.numDimensions))

        if k > self.numItems:
            k = self.numItems

        cdef np.ndarray[DATA_TYPE_C, ndim=2] distances = np.empty((numQueries, k), dtype=NUMPY_DATA_TYPE, order="c")
        cdef np.ndarray[INDEX_TYPE_C, ndim=2] idx_array = np.empty((numQueries, k), dtype=NUMPY_INDEX_TYPE, order="c")

        # initialize arrays.  This is only needed for correct behavior of
        # the warning flag.
        idx_array[:] = -9999
        distances[:] = np.inf

        self.__query_many(<DATA_TYPE_C*>queryData.data, numQueries, k, numDimensions,
                          <DATA_TYPE_C*>distances.data, <INDEX_TYPE_C*>idx_array.data)

        return distances, idx_array


    cdef void __query_many(BallTree self,
                           DATA_TYPE_C* queryData,
                           INDEX_TYPE_C numQueries,
                           INDEX_TYPE_C k,
                           INDEX_TYPE_C numDimensions,
                           DATA_TYPE_C* distances,
                           INDEX_TYPE_C* idx_array) nogil:

        cdef DATA_TYPE_C* query_ptr
        cdef DATA_TYPE_C* dist_ptr
        cdef INDEX_TYPE_C* idx_ptr

        cdef stack *node_stacks

        cdef int i

        with nogil, parallel():
            node_stacks = <stack*> malloc(sizeof(stack) * numQueries)

            # query using pointers
            # for k less than 5, a priority queue is slightly faster
            # for more neighbors, a max-heap implementation is faster
            if k >= 5:
                for i in prange(numQueries):
                    stack_create(&node_stacks[i], self.n_levels + 1)
                    query_ptr = queryData + i*numDimensions
                    dist_ptr= distances + i*k
                    idx_ptr = idx_array + i*k

                    self.query_one_max_heap(query_ptr, k, dist_ptr, idx_ptr, &node_stacks[i])
                    # if max-heap is used, results must be sorted
                    sort_dist_idx(dist_ptr, idx_ptr, k)
                    stack_destroy(&node_stacks[i])

            else:
                for i in prange(numQueries):
                    stack_create(&node_stacks[i], self.n_levels + 1)
                    query_ptr = queryData + i*numDimensions
                    dist_ptr= distances + i*k
                    idx_ptr = idx_array + i*k

                    self.query_one_pq(query_ptr, k, dist_ptr, idx_ptr, &node_stacks[i])
                    stack_destroy(&node_stacks[i])

            free(node_stacks)

    def query_radius(self, X, DATA_TYPE_C radius, count_only=False):
        """
        query_radius(self, X, r, count_only = False):

        query the Ball Tree for neighbors within a ball of size r

        Parameters
        ----------
        X : array-like, last dimension self.dim
            An array of points to query
        radius : distance within which neighbors are returned
        count_only : boolean (default = False)
            if True,  return only the count of points within distance r
            if False, return the indices of all points within distance r

        Returns
        -------
        count  : if count_only == True
        (d, i) : if count_only == False

        count : array of integers, shape = X.shape[:-1]
            each entry gives the number of neighbors within
            a distance r of the corresponding point.

        d : array of floats - the list of distances to the
            neighbors of the corresponding point
            (note that distances are not sorted)

        i : array of integers the list of indices of
            neighbors of the corresponding point
            (note that neighbors are not sorted)

        Examples
        --------
        Query for neighbors in a given radius

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        >>> ball_tree = BallTree(X, leaf_size=2)        # doctest: +SKIP
        >>> print ball_tree.query_radius(X[0], r=0.3, count_only=True)
        3
        >>> ind = ball_tree.query_radius(X[0], r=0.3)  # doctest: +SKIP
        >>> print ind  # indices of neighbors within distance 0.3
        [3 0 1]
        """
        cdef INDEX_TYPE_C count
        # prepare X for query
        if len(X) != self.numDimensions:
            raise ValueError("query data dimension must match BallTree data dimension")

        cdef np.ndarray[DATA_TYPE_C, ndim=1] queryData = np.asarray(X, dtype=NUMPY_DATA_TYPE)

        cdef stack node_stack
        stack_create(&node_stack, self.n_levels + 1)

        if count_only:
            count = self.query_radius_count_(<DATA_TYPE_C*>queryData.data, radius, &node_stack)
            stack_destroy(&node_stack)
            return count

        cdef np.ndarray[INDEX_TYPE_C, ndim=1] idx_array = np.empty(len(self.data), dtype=NUMPY_INDEX_TYPE)
        cdef np.ndarray[DATA_TYPE_C, ndim=1] distances = np.empty(len(self.data), dtype=NUMPY_DATA_TYPE)

        count = self.query_radius_distances_(<DATA_TYPE_C*>queryData.data,
                                             radius,
                                             <INDEX_TYPE_C*>idx_array.data,
                                             <DATA_TYPE_C*>distances.data,
                                             &node_stack)
        stack_destroy(&node_stack)
        return distances[:count], idx_array[:count]

    cdef void build_tree_(BallTree self):
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function

        cdef INDEX_TYPE_C n_samples = self.numItems
        cdef INDEX_TYPE_C n_features = self.numDimensions

        cdef INDEX_TYPE_C idx_start, idx_end, n_points
        cdef DATA_TYPE_C radius
        cdef INDEX_TYPE_C i, i_node, i_parent

        cdef DATA_TYPE_C* centroid = node_centroid_arr
        cdef NodeInfo* node_info = node_info_arr
        cdef NodeInfo* parent_info
        cdef DATA_TYPE_C* point

        #------------------------------------------------------------
        # take care of the root node
        node_info.idx_start = 0
        node_info.idx_end = n_samples
        n_points = n_samples

        # determine Node centroid
        compute_centroid(centroid, data, idx_array,
                         n_features, n_samples)

        # determine Node radius
        radius = 0
        for i in range(node_info.idx_start, node_info.idx_end):
            radius = dmax(radius,
                          distance_function(centroid, data + n_features * idx_array[i], n_features, p))
        node_info.radius = radius

        # check if this is a leaf
        if self.n_nodes == 1:
            node_info.is_leaf = 1

        else:
            node_info.is_leaf = 0

            # find dimension with largest spread
            i_max = find_split_dim(data, idx_array + node_info.idx_start, n_features, n_points)

            # sort idx_array along this dimension
            partition_indices(data,
                              idx_array + node_info.idx_start,
                              i_max,
                              n_points / 2,
                              n_features,
                              n_points)

        #------------------------------------------------------------
        # cycle through all child nodes
        for i_node in range(1, self.n_nodes):
            i_parent = (i_node - 1) / 2
            parent_info = node_info_arr + i_parent

            node_info = node_info_arr + i_node

            if parent_info.is_leaf:
                raise ValueError("Fatal: parent is a leaf. Memory "
                                 "allocation is flawed")

            if i_node < self.n_nodes / 2:
                node_info.is_leaf = 0
            else:
                node_info.is_leaf = 1

            centroid = node_centroid_arr + i_node * n_features

            # find indices for this node
            idx_start = parent_info.idx_start
            idx_end = parent_info.idx_end

            if i_node % 2 == 1:
                idx_start = (idx_start + idx_end) / 2
            else:
                idx_end = (idx_start + idx_end) / 2

            node_info.idx_start = idx_start
            node_info.idx_end = idx_end

            n_points = idx_end - idx_start

            if n_points == 0:
                raise ValueError("zero-sized node")

            elif n_points == 1:
                #copy this point to centroid
                copy_array(centroid,
                           data + idx_array[idx_start] * n_features,
                           n_features)

                #store radius in array
                node_info.radius = 0

                #is a leaf
                node_info.is_leaf = 1

            else:
                # determine Node centroid
                compute_centroid(centroid, data, idx_array + idx_start,
                                 n_features, n_points)

                # determine Node radius
                radius = 0
                for i in range(idx_start, idx_end):
                    radius = dmax(radius,
                                  distance_function(centroid, data + n_features * idx_array[i], n_features, p))
                node_info.radius = radius

                if not node_info.is_leaf:
                    # find dimension with largest spread
                    i_max = find_split_dim(data, idx_array + idx_start, n_features, n_points)

                    # sort indices along this dimension
                    partition_indices(data,
                                      idx_array + idx_start,
                                      i_max,
                                      n_points / 2,
                                      n_features,
                                      n_points)

    cdef void query_one_(BallTree self,
                         DATA_TYPE_C* pt,
                         INDEX_TYPE_C k,
                         DATA_TYPE_C* near_set_dist,
                         INDEX_TYPE_C* near_set_indx,
                         stack* node_stack,
                         INDEX_TYPE_C use_max_heap) nogil:
        if use_max_heap:
            self.query_one_max_heap(pt, k, near_set_dist, near_set_indx, node_stack)
        else:
            self.query_one_pq(pt, k, near_set_dist, near_set_indx, node_stack)

    cdef void query_one_pq(BallTree self,
                           DATA_TYPE_C* pt,
                           INDEX_TYPE_C k,
                           DATA_TYPE_C* near_set_dist,
                           INDEX_TYPE_C* near_set_indx,
                           stack* node_stack) nogil:
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function
        cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        pseudo_distance_function = self.pseudo_distance_function

        cdef INDEX_TYPE_C n_features = self.numDimensions

        cdef DATA_TYPE_C maxDistance, dist_pt, dist_p_LB, dist_p_LB_1, dist_p_LB_2
        cdef INDEX_TYPE_C i, i1, i2, i_node

        cdef stack_item item

        # This will keep track of any indices with distances values.  If at
        # the end of the tree traversal, this index is in the last position,
        # then the warning flag will be set.
        cdef INDEX_TYPE_C check_index = -1

        item.i_node = 0
        item.dist_p_LB = lower_bound_pseudo_distance(pt, node_centroid_arr,
                                                     node_info.radius,
                                                     n_features, p, distance_function)
        stack_push(node_stack, item)

        while node_stack.n > 0:
            item = stack_pop(node_stack)
            i_node = item.i_node
            dist_p_LB = item.dist_p_LB

            node_info = node_info_arr + i_node

            #------------------------------------------------------------
            # Case 0: query point is exactly on the boundary.  Set
            #         warning flag
            if dist_p_LB == pqueue_largest(near_set_dist, k):
                # store index of point with same distance:
                # we'll check it later
                check_index = pqueue_idx_largest(near_set_indx, k)
                continue

            #------------------------------------------------------------
            # Case 1: query point is outside node radius
            elif dist_p_LB > pqueue_largest(near_set_dist, k):
                continue

            #------------------------------------------------------------
            # Case 2: this is a leaf node.  Update set of nearby points
            elif node_info.is_leaf:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt,
                                     data + n_features * idx_array[i],
                                     n_features, p)

                    maxDistance = pqueue_largest(near_set_dist, k)

                    if dist_pt == maxDistance:
                        check_index = pqueue_idx_largest(near_set_indx, k)

                    elif dist_pt < maxDistance:
                        pqueue_insert(dist_pt, idx_array[i],
                                         near_set_dist, near_set_indx, k)
                        if maxDistance == pqueue_largest(near_set_dist, k):
                            check_index = pqueue_idx_largest(near_set_indx,
                                                                k)

            #------------------------------------------------------------
            # Case 3: Node is not a leaf.  Recursively query subnodes
            #         starting with the one whose centroid is closest
            else:
                i1 = 2 * i_node + 1
                i2 = i1 + 1
                dist_p_LB_1 = lower_bound_pseudo_distance(pt, (node_centroid_arr + i1 * n_features),
                                                          node_info_arr[i1].radius,
                                                          n_features, p, distance_function)
                dist_p_LB_2 = lower_bound_pseudo_distance(pt, (node_centroid_arr + i2 * n_features),
                                                          node_info_arr[i2].radius,
                                                          n_features, p, distance_function)

                # append children to stack: last-in-first-out
                if dist_p_LB_2 <= dist_p_LB_1:
                    item.i_node = i1
                    item.dist_p_LB = dist_p_LB_1
                    stack_push(node_stack, item)

                    item.i_node = i2
                    item.dist_p_LB = dist_p_LB_2
                    stack_push(node_stack, item)

                else:
                    item.i_node = i2
                    item.dist_p_LB = dist_p_LB_2
                    stack_push(node_stack, item)

                    item.i_node = i1
                    item.dist_p_LB = dist_p_LB_1
                    stack_push(node_stack, item)

        # remove conversion between dist/dist_p
        #for i from 0 <= i < k:
        #    near_set_dist[i] = dist_from_dist_p(near_set_dist[i], p)


    cdef void query_one_max_heap(BallTree self,
                                 DATA_TYPE_C* pt,
                                 INDEX_TYPE_C k,
                                 DATA_TYPE_C* near_set_dist,
                                 INDEX_TYPE_C* near_set_indx,
                                 stack* node_stack) nogil:
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function
        cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        pseudo_distance_function = self.pseudo_distance_function

        cdef INDEX_TYPE_C n_features = self.numDimensions

        cdef DATA_TYPE_C max_distance, dist_pt, dist_p_LB, dist_p_LB_1, dist_p_LB_2
        cdef INDEX_TYPE_C i, i1, i2, i_node

        cdef stack_item item

        # This will keep track of any indices with distances values.  If at
        # the end of the tree traversal, this index is in the last position,
        # then the warning flag will be set.
        cdef INDEX_TYPE_C check_index = -1

        item.i_node = 0
        item.dist_p_LB = lower_bound_pseudo_distance(pt, node_centroid_arr, node_info.radius,
                                                     n_features, p, distance_function)
        stack_push(node_stack, item)

        while node_stack.n > 0:
            item = stack_pop(node_stack)
            i_node = item.i_node
            dist_p_LB = item.dist_p_LB

            node_info = node_info_arr + i_node

            #------------------------------------------------------------
            # Case 0: query point is exactly on the boundary.  Set
            #         warning flag
            if dist_p_LB == max_heap_largest(near_set_dist, k):
                # store index of point with same distance:
                # we'll check it later
                check_index = max_heap_idx_largest(near_set_indx, k)
                continue

            #------------------------------------------------------------
            # Case 1: query point is outside node radius
            elif dist_p_LB > max_heap_largest(near_set_dist, k):
                continue

            #------------------------------------------------------------
            # Case 2: this is a leaf node.  Update set of nearby points
            elif node_info.is_leaf:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt, data + n_features * idx_array[i], n_features, p)

                    max_distance = max_heap_largest(near_set_dist, k)

                    if dist_pt == max_distance:
                        check_index = max_heap_idx_largest(near_set_indx, k)

                    elif dist_pt < max_distance:
                        max_heap_insert(dist_pt, idx_array[i],
                                         near_set_dist, near_set_indx, k)
                        if max_distance == max_heap_largest(near_set_dist, k):
                            check_index = max_heap_idx_largest(near_set_indx,
                                                                k)

            #------------------------------------------------------------
            # Case 3: Node is not a leaf.  Recursively query subnodes
            #         starting with the one whose centroid is closest
            else:
                i1 = 2 * i_node + 1
                i2 = i1 + 1
                dist_p_LB_1 = lower_bound_pseudo_distance(pt, (node_centroid_arr + i1 * n_features),
                                                          node_info_arr[i1].radius,
                                                          n_features, p, distance_function)
                dist_p_LB_2 = lower_bound_pseudo_distance(pt, (node_centroid_arr + i2 * n_features),
                                                          node_info_arr[i2].radius,
                                                          n_features, p, distance_function)

                # append children to stack: last-in-first-out
                if dist_p_LB_2 <= dist_p_LB_1:
                    item.i_node = i1
                    item.dist_p_LB = dist_p_LB_1
                    stack_push(node_stack, item)

                    item.i_node = i2
                    item.dist_p_LB = dist_p_LB_2
                    stack_push(node_stack, item)

                else:
                    item.i_node = i2
                    item.dist_p_LB = dist_p_LB_2
                    stack_push(node_stack, item)

                    item.i_node = i1
                    item.dist_p_LB = dist_p_LB_1
                    stack_push(node_stack, item)

        # remove conversion between dist/dist_p
        #for i from 0 <= i < k:
        #    near_set_dist[i] = dist_from_dist_p(near_set_dist[i], p)

    cdef INDEX_TYPE_C query_radius_count_(BallTree self,
                                     DATA_TYPE_C* pt, DATA_TYPE_C r,
                                     stack* node_stack) nogil:
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function
        cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        pseudo_distance_function = self.pseudo_distance_function
        cdef INDEX_TYPE_C n_features = self.numDimensions
        cdef INDEX_TYPE_C i, i_node
        cdef INDEX_TYPE_C count = 0
        cdef DATA_TYPE_C r_p = dist_p_from_dist(r, p)
        cdef DATA_TYPE_C dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while node_stack.n > 0:
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = distance_function(pt, node_centroid_arr + n_features * i_node,
                           n_features, p)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r_p:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r_p:
                count += (node_info.idx_end - node_info.idx_start)

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt, data + idx_array[i] * n_features, n_features, p)
                    if dist_pt <= r_p:
                        count += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return count

    cdef INDEX_TYPE_C query_radius_idx_only_(BallTree self,
                                        DATA_TYPE_C* pt, DATA_TYPE_C r,
                                        INDEX_TYPE_C* indices,
                                        stack* node_stack) nogil:
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function
        cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        pseudo_distance_function = self.pseudo_distance_function
        cdef INDEX_TYPE_C n_features = self.numDimensions
        cdef INDEX_TYPE_C i, i_node
        cdef INDEX_TYPE_C idx_i = 0
        cdef DATA_TYPE_C r_p = dist_p_from_dist(r, p)
        cdef DATA_TYPE_C dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while node_stack.n > 0:
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = distance_function(pt, node_centroid_arr + n_features * i_node,
                           n_features, p)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r_p:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r_p:
                for i in range(node_info.idx_start, node_info.idx_end):
                    indices[idx_i] = idx_array[i]
                    idx_i += 1

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt, data + idx_array[i] * n_features, n_features, p)
                    if dist_pt <= r_p:
                        indices[idx_i] = idx_array[i]
                        idx_i += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return idx_i

    cdef INDEX_TYPE_C query_radius_distances_(BallTree self,
                                         DATA_TYPE_C* pt, DATA_TYPE_C r,
                                         INDEX_TYPE_C* indices,
                                         DATA_TYPE_C* distances,
                                         stack* node_stack) nogil:
        cdef DATA_TYPE_C* data = <DATA_TYPE_C*> self.data.data
        cdef INDEX_TYPE_C* idx_array = <INDEX_TYPE_C*> self.idx_array.data
        cdef DATA_TYPE_C* node_centroid_arr = <DATA_TYPE_C*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef DATA_TYPE_C p = self.p
        cdef DATA_TYPE_C (*distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        distance_function = self.distance_function
        cdef DATA_TYPE_C (*pseudo_distance_function)(DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil
        pseudo_distance_function = self.pseudo_distance_function

        cdef INDEX_TYPE_C n_features = self.numDimensions
        cdef INDEX_TYPE_C i, i_node
        cdef INDEX_TYPE_C idx_i = 0
        cdef DATA_TYPE_C r_p = dist_p_from_dist(r, p)
        cdef DATA_TYPE_C dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while node_stack.n > 0:
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = distance_function(pt, node_centroid_arr + n_features * i_node,
                           n_features, p)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r_p:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r_p:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt,
                                   data + idx_array[i] * n_features,
                                   n_features, p)
                    indices[idx_i] = idx_array[i]
                    distances[idx_i] = dist_pt
                    idx_i += 1

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i in range(node_info.idx_start, node_info.idx_end):
                    dist_pt = pseudo_distance_function(pt,
                                     data + idx_array[i] * n_features,
                                     n_features, p)
                    if dist_pt <= r_p:
                        indices[idx_i] = idx_array[i]
                        distances[idx_i] = dist_pt
                        idx_i += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return idx_i


######################################################################
# Helper functions for building and querying
######################################################################
cdef inline void copy_array(DATA_TYPE_C* x, DATA_TYPE_C* y, INDEX_TYPE_C n) nogil:
    # copy array y into array x
    cdef INDEX_TYPE_C i
    for i in range(n):
        x[i] = y[i]


cdef void compute_centroid(DATA_TYPE_C* centroid,
                           DATA_TYPE_C* data,
                           INDEX_TYPE_C* node_indices,
                           INDEX_TYPE_C n_features,
                           INDEX_TYPE_C n_points) nogil:
    # `centroid` points to an array of length n_features
    # `data` points to an array of length n_samples * n_features
    # `node_indices` = idx_array + idx_start
    cdef DATA_TYPE_C *this_pt
    cdef INDEX_TYPE_C i, j

    for j in range(n_features):
        centroid[j] = 0

    for i in range(n_points):
        this_pt = data + n_features * node_indices[i]
        for j in range(n_features):
            centroid[j] += this_pt[j]

    for j in range(n_features):
        centroid[j] /= n_points


cdef INDEX_TYPE_C find_split_dim(DATA_TYPE_C* data,
                            INDEX_TYPE_C* node_indices,
                            INDEX_TYPE_C n_features,
                            INDEX_TYPE_C n_points) nogil:
    # this computes the following
    # j_max = np.argmax(np.max(data, 0) - np.min(data, 0))
    cdef DATA_TYPE_C min_val, max_val, val, spread, max_spread
    cdef INDEX_TYPE_C i, j, j_max

    j_max = 0
    max_spread = 0

    for j in range(n_features):
        max_val = data[node_indices[0] * n_features + j]
        min_val = max_val
        for i in range(n_points):
            val = data[node_indices[i] * n_features + j]
            max_val = dmax(max_val, val)
            min_val = dmin(min_val, val)
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max

cdef inline void iswap(INDEX_TYPE_C* arr, INDEX_TYPE_C i1, INDEX_TYPE_C i2) nogil:
    cdef INDEX_TYPE_C tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp

cdef inline void dswap(DATA_TYPE_C* arr, INDEX_TYPE_C i1, INDEX_TYPE_C i2) nogil:
    cdef DATA_TYPE_C tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


cdef void partition_indices(DATA_TYPE_C* data,
                            INDEX_TYPE_C* node_indices,
                            INDEX_TYPE_C split_dim,
                            INDEX_TYPE_C split_index,
                            INDEX_TYPE_C n_features,
                            INDEX_TYPE_C n_points) nogil:
    # partition_indices will modify the array node_indices between
    # indices 0 and n_points.  Upon return (assuming numpy-style slicing)
    #   data[node_indices[0:split_index], split_dim]
    #     <= data[node_indices[split_index], split_dim]
    # and
    #   data[node_indices[split_index], split_dim]
    #     <= data[node_indices[split_index:n_points], split_dim]
    # will hold.  The algorithm amounts to a partial quicksort
    cdef INDEX_TYPE_C left, right, midindex, i
    cdef DATA_TYPE_C d1, d2
    left = 0
    right = n_points - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[node_indices[i] * n_features + split_dim]
            d2 = data[node_indices[right] * n_features + split_dim]
            if d1 < d2:
                iswap(node_indices, i, midindex)
                midindex += 1
        iswap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


######################################################################
# calc_dist_p_LB
#  This calculates the lower-bound distance between a point and a node
######################################################################
cdef inline DATA_TYPE_C lower_bound_pseudo_distance(DATA_TYPE_C* pt,
                                                    DATA_TYPE_C* centroid,
                                                    DATA_TYPE_C radius,
                                                    INDEX_TYPE_C n_features,
                                                    DATA_TYPE_C p,
                                                    DATA_TYPE_C (*distance_function)
                                                     (DATA_TYPE_C*, DATA_TYPE_C*, INDEX_TYPE_C, DATA_TYPE_C) nogil) nogil:
    return dist_p_from_dist(dmax(0, (distance_function(pt, centroid, n_features, p) - radius)), p)


######################################################################
# priority queue
#  This is used to keep track of the neighbors as they are found.
#  It keeps the list of neighbors sorted, and inserts each new item
#  into the list.  In this fixed-size implementation, empty elements
#  are represented by infinities.
######################################################################
cdef inline DATA_TYPE_C pqueue_largest(DATA_TYPE_C* queue, INDEX_TYPE_C queue_size) nogil:
    return queue[queue_size - 1]


cdef inline INDEX_TYPE_C pqueue_idx_largest(INDEX_TYPE_C* idx_array, INDEX_TYPE_C queue_size) nogil:
    return idx_array[queue_size - 1]


cdef inline void pqueue_insert(DATA_TYPE_C val, INDEX_TYPE_C i_val,
                               DATA_TYPE_C* queue, INDEX_TYPE_C* idx_array,
                               INDEX_TYPE_C queue_size) nogil:
    cdef INDEX_TYPE_C i_lower = 0
    cdef INDEX_TYPE_C i_upper = queue_size - 1
    cdef INDEX_TYPE_C i_mid
    cdef INDEX_TYPE_C i

    if val >= queue[i_upper]:
        return
    elif val <= queue[i_lower]:
        i_mid = i_lower
    else:
        while True:
            if (i_upper - i_lower) < 2:
                i_mid = i_lower + 1
                break
            else:
                i_mid = (i_lower + i_upper) / 2

            if i_mid == i_lower:
                i_mid += 1
                break

            if val >= queue[i_mid]:
                i_lower = i_mid
            else:
                i_upper = i_mid

    for i in range(queue_size-1, i_mid, -1):
        queue[i] = queue[i - 1]
        idx_array[i] = idx_array[i - 1]

    queue[i_mid] = val
    idx_array[i_mid] = i_val


######################################################################
# max_heap
#
#  This is a basic implementation of a fixed-size binary max-heap.
#  It can be used in place of priority_queue to keep track of the
#  k-nearest neighbors in a query.  The implementation is faster than
#  priority_queue for a very large number of neighbors (k > 50 or so).
#  The implementation is slower than priority_queue for fewer neighbors.
#  The other disadvantage is that for max_heap, the indices/distances must
#  be sorted upon completion of the query.  In priority_queue, the indices
#  and distances are sorted without an extra call.
#
#  The root node is at heap[0].  The two child nodes of node i are at
#  (2 * i + 1) and (2 * i + 2).
#  The parent node of node i is node floor((i-1)/2).  Node 0 has no parent.
#  A max heap has (heap[i] >= heap[2 * i + 1]) and (heap[i] >= heap[2 * i + 2])
#  for all valid indices.
#
#  In this implementation, an empty heap should be full of infinities
#
#  As part of this implementation, there is a quicksort provided with
#  `sort_dist_idx()`
######################################################################
cdef inline DATA_TYPE_C max_heap_largest(DATA_TYPE_C* heap, INDEX_TYPE_C k) nogil:
    return heap[0]


cdef inline INDEX_TYPE_C max_heap_idx_largest(INDEX_TYPE_C* idx_array, INDEX_TYPE_C k) nogil:
    return idx_array[0]


cdef void max_heap_insert(DATA_TYPE_C val, INDEX_TYPE_C i_val,
                          DATA_TYPE_C* heap,
                          INDEX_TYPE_C* idx_array,
                          INDEX_TYPE_C heap_size) nogil:
    cdef INDEX_TYPE_C i, ic1, ic2, i_tmp
    cdef DATA_TYPE_C d_tmp

    # check if val should be in heap
    if val > heap[0]:
        return

    # insert val at position zero
    heap[0] = val
    idx_array[0] = i_val

    #descend the heap, swapping values until the max heap criterion is met
    i = 0
    while 1:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap_size:
            break
        elif ic2 >= heap_size:
            if heap[ic1] > val:
                i_swap = ic1
            else:
                break
        elif heap[ic1] >= heap[ic2]:
            if val < heap[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < heap[ic2]:
                i_swap = ic2
            else:
                break

        heap[i] = heap[i_swap]
        idx_array[i] = idx_array[i_swap]

        i = i_swap

    heap[i] = val
    idx_array[i] = i_val


######################################################################
# sort_dist_idx :
#  this is a quicksort implementation which sorts `dist` and
#  simultaneously performs the same swaps on `idx`.
######################################################################
cdef void sort_dist_idx(DATA_TYPE_C* dist, INDEX_TYPE_C* idx, INDEX_TYPE_C k) nogil:
    cdef INDEX_TYPE_C pivot_idx
    if k > 1:
        pivot_idx = partition_dist_idx(dist, idx, k)

        sort_dist_idx(dist, idx, pivot_idx)

        sort_dist_idx(dist + pivot_idx + 1,
                      idx + pivot_idx + 1,
                      k - pivot_idx - 1)


cdef INDEX_TYPE_C partition_dist_idx(DATA_TYPE_C* dist, INDEX_TYPE_C* idx, INDEX_TYPE_C k) nogil:
    cdef INDEX_TYPE_C pivot_idx = k / 2
    cdef DATA_TYPE_C pivot_val = dist[pivot_idx]
    cdef INDEX_TYPE_C store_idx = 0
    cdef INDEX_TYPE_C i

    dswap(dist, pivot_idx, k - 1)
    iswap(idx, pivot_idx, k - 1)

    for i in range(k - 1):
        if dist[i] < pivot_val:
            dswap(dist, i, store_idx)
            iswap(idx, i, store_idx)
            store_idx += 1
    dswap(dist, store_idx, k - 1)
    iswap(idx, store_idx, k - 1)
    return store_idx