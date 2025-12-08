"""K-means clustering."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.cluster._k_means_common import (
    CHUNK_SIZE,
    _inertia_dense,
    _inertia_sparse,
    _is_same_clustering,
)
from sklearn.preprocessing import normalize
from sklearn.cluster._k_means_elkan import (
    elkan_iter_chunked_dense,
    elkan_iter_chunked_sparse,
    init_bounds_dense,
    init_bounds_sparse,
)
from sklearn.cluster._k_means_lloyd import (
    lloyd_iter_chunked_dense,
    lloyd_iter_chunked_sparse,
)
from sklearn.cluster._k_means_minibatch import (
    _minibatch_update_dense,
    _minibatch_update_sparse,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import _euclidean_distances, euclidean_distances, cosine_distances, manhattan_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import row_norms
from sklearn.utils.parallel import (
    _get_threadpool_controller,
    _threadpool_controller_decorator,
)
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_is_fitted,
    validate_data,
)


def _compute_distances(
    X, Y, metric, Y_norm_squared=None, squared=False
):
    """Compute the distance between each pair of the two collections of inputs.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples_1, n_features)
        First input.

    Y : {ndarray, sparse matrix} of shape (n_samples_2, n_features)
        Second input.

    metric : {"euclidean", "cosine", "manhattan"}
        The metric to use when calculating distance between points.

    Y_norm_squared : ndarray of shape (n_samples_2,), default=None
        Precomputed squared euclidean norm of each row in Y.

    squared : bool, default=False
        If True, return squared Euclidean distances.



    Returns
    -------
    distances : ndarray of shape (n_samples_1, n_samples_2)
        Distances between each pair of the two collections of inputs.
    """
    if metric == "euclidean":
        return _euclidean_distances(
            X, Y, Y_norm_squared=Y_norm_squared, squared=squared
        )
    elif metric == "cosine":
        distance = cosine_distances(X, Y)
        if squared:
            distance **= 2
        return distance
    elif metric == "manhattan":
        distance = manhattan_distances(X, Y)
        if squared:
            distance **= 2
        return distance
    else:
        raise ValueError(f"Unknown metric '{metric}'")

###############################################################################
# Initialization heuristic


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "cosine", "manhattan"})],
        "sample_weight": ["array-like", None],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def kmeans_plusplus(
    X,
    n_clusters,
    metric,
    *,
    sample_weight=None,
    x_squared_norms=None,
    random_state=None,
    n_local_trials=None,
):
    """Init n_clusters seeds according to k-means++.

    .. versionadded:: 0.24

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.

    n_clusters : int
        The number of centroids to initialize.

    metric : {"euclidean", "cosine", "manhattan"}
        The metric to use when calculating distance between points.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is ignored if `init`
        is a callable or a user provided array.

        .. versionadded:: 1.3

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.

    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)) which is the recommended setting.
        Setting to 1 disables the greedy cluster selection and recovers the
        vanilla k-means++ algorithm which was empirically shown to work less
        well than its greedy variant.



    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  2],
           [ 1,  0]])
    >>> indices
    array([3, 2])
    """
    # Check data
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    random_state = check_random_state(random_state)

    # Call private k-means++
    centers, indices = _kmeans_plusplus(
        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials, metric=metric,
    )

    return centers, indices


def _kmeans_plusplus(
    X, n_clusters, x_squared_norms, sample_weight, random_state, metric, n_local_trials=None
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    metric : {"euclidean", "cosine", "manhattan"}, default="euclidean"
        The metric to use when calculating distance between points.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _compute_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True, metric=metric,
    )
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _compute_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True, metric=metric
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
        "return_n_iter": [bool],
    },
    prefer_skip_nested_validation=False,
)
def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    metric="euclidean",
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    """Perform K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is not used during
        initialization if `init` is a callable or a user provided array.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        - `'k-means++'` : selects initial cluster centers for k-mean
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.
        - `'random'`: choose `n_clusters` observations (rows) at random from data
          for the initial centroids.
        - If an array is passed, it should be of shape `(n_clusters, n_features)`
          and gives the initial centers.
        - If a callable is passed, it should take arguments `X`, `n_clusters` and a
          random state and return an initialization.

    n_init : 'auto' or int, default="auto"
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If `copy_x` is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        `copy_x` is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if `copy_x` is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        The `label[i]` is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import k_means
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centroid, label, inertia = k_means(
    ...     X, n_clusters=2, n_init="auto", random_state=0
    ... )
    >>> centroid
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> label
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> inertia
    16.0
    """
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        metric=metric,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_


def _kmeans_single_elkan(
    X,
    sample_weight,
    centers_init,
    metric,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means elkan, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    metric : {"euclidean", "cosine", "manhattan"}, default="euclidean"
        The metric to use when calculating distance between points.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = _compute_distances(centers, centers, metric=metric) / 2
    distance_next_center = np.partition(
        np.asarray(center_half_distances), kth=1, axis=0
    )[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(
        X,
        centers,
        center_half_distances,
        labels,
        upper_bounds,
        lower_bounds,
        n_threads=n_threads,
    )

    strict_convergence = False

    for i in range(max_iter):
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
        )

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = _compute_distances(centers_new, centers_new, metric=metric) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0
        )[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


# Threadpoolctl context to limit the number of threads in second level of
# nested parallelism (i.e. BLAS) to avoid oversubscription.
@_threadpool_controller_decorator(limits=1, user_api="blas")
def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    metric,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    metric : {"euclidean", "cosine", "manhattan"}, default="euclidean"
        The metric to use when calculating distance between points.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    for i in range(max_iter):
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
        )

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}.")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)

    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    return labels


# Same as _labels_inertia but in a threadpool_limits context.
_labels_inertia_threadpool_limit = _threadpool_controller_decorator(
    limits=1, user_api="blas"
)(_labels_inertia)


class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):
    """Base class for KMeans and MiniBatchKMeans"""

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "metric": [StrOptions({"euclidean", "cosine", "manhattan"})],
    }

    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        metric,
        tol,
        verbose,
        random_state,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.metric = metric
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_input(self, X, default_n_init=None):
        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = _tolerance(X, self.tol)

        # n-init
        if self.n_init == "auto":
            if isinstance(self.init, str) and self.init == "k-means++":
                self._n_init = 1
            elif isinstance(self.init, str) and self.init == "random":
                self._n_init = default_n_init
            elif callable(self.init):
                self._n_init = default_n_init
            else:  # array-like
                self._n_init = 1
        else:
            self._n_init = self.n_init

        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.__class__.__name__} instead of "
                    f"n_init={self._n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        """Issue an estimator specific warning when vcomp and mkl are both present

        This method is called by `_check_mkl_vcomp`.
        """

    def _check_mkl_vcomp(self, X, n_samples):
        """Check when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if n_active_threads < self._n_threads:
            modules = _get_threadpool_controller().info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        sample_weight,
        init_size=None,
        n_centroids=None,
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
                metric=self.metric,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)

        # sample weights are not used by predict but cython helpers expect an array
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

        if self.metric == "euclidean":
            # Use the more efficient euclidean specific implementation
            labels, _ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
        else:
            labels = np.argmin(
                _compute_distances(X, self.cluster_centers_, metric=self.metric), axis=1
            )

        return labels

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return _compute_distances(X, self.cluster_centers_, metric=self.metric)

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if self.metric != "euclidean":
            raise NotImplementedError(
                "score method is only implemented for "
                "metric='euclidean' got "
                f"metric='{self.metric}' instead."
            )
        _, scores = _labels_inertia_threadpool_limit(
            X, sample_weight, self.cluster_centers_, self._n_threads
        )
        return -scores

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class KMeans(_BaseKMeans):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling \
            based on an empirical probability distribution of the points' \
            contribution to the overall inertia. This technique speeds up \
            convergence. The algorithm implemented is "greedy k-means++". It \
            differs from the vanilla k-means++ by making several trials at \
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from \
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)\
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a\
        random state and return an initialization.

        For an example of how to use the different `init` strategies, see
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

        For an evaluation of the impact of initialization, see the example
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_stability_low_dim_dense.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    metric : {"euclidean", "cosine", "manhattan"}, default="euclidean"
        The metric to use when calculating distance between points.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])

    For examples of common problems with K-Means and how to address them see
    :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`.

    For a demonstration of how K-Means can be used to cluster text documents see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

    For a comparison between K-Means and MiniBatchKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`.

    For a comparison between K-Means and BisectingKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_bisect_kmeans.py`.
    """

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd", "elkan"})],
        "metric": [StrOptions({"euclidean", "cosine", "manhattan"})],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        metric="euclidean",
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            metric=metric,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params_vs_input(self, X):
        super()._check_params_vs_input(X, default_n_init=10)

        self._algorithm = self.algorithm
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                (
                    "algorithm='elkan' doesn't make sense for a single "
                    "cluster. Using 'lloyd' instead."
                ),
                RuntimeWarning,
            )
            self._algorithm = "lloyd"

        if self.algorithm == "elkan" and self.metric != "euclidean":
            raise ValueError(
                "Elkan algorithm only works with 'euclidean' metric. "
                f"Got metric='{self.metric}'."
            )

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if init_is_array_like:
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "elkan":
            kmeans_single = _kmeans_single_elkan
        else:
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                sample_weight=sample_weight,
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                n_threads=self._n_threads,
                metric=self.metric,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        # If the metric is not euclidean, we need to recompute the labels
        # for the best centers found
        if self.metric != "euclidean":
            best_labels = np.argmin(
                _compute_distances(X, best_centers, metric=self.metric), axis=1
            )

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
