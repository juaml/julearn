from sklearn.cluster import KMeans as SklearnKMeans
from julearn.external.kmeans import KMeans as JulearnKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
import numpy as np
from numpy.testing import assert_array_equal


def test_kmeans_equivalence():
    X = [[1, 2], [1, 4], [1, 0],
         [4, 2], [4, 4], [4, 0]]
    n_clusters = 2
    random_state = 42

    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=random_state)
    julearn_kmeans = JulearnKMeans(n_clusters=n_clusters, random_state=random_state)

    sklearn_kmeans.fit(X)
    julearn_kmeans.fit(X)

    assert (sklearn_kmeans.cluster_centers_ == julearn_kmeans.cluster_centers_).all()
    assert_array_equal(sklearn_kmeans.labels_, julearn_kmeans.labels_)
    assert sklearn_kmeans.inertia_ == julearn_kmeans.inertia_


def test_kmeans_cosine_metric():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]], dtype=np.float32)
    random_state = 42
    n_clusters = 2

    print("Sklearn KMeans cluster centers (normalized)")
    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=random_state, verbose=199)
    X_norm = normalize(X, norm='l2')
    sklearn_kmeans.fit(X_norm)

    print("Julearn KMeans cluster centers (cosine metric)")
    julearn_kmeans = JulearnKMeans(n_clusters=n_clusters, random_state=random_state, metric="cosine", verbose=199)
    julearn_kmeans.fit(X)

    assert julearn_kmeans.metric == "cosine"
    assert len(julearn_kmeans.cluster_centers_) == n_clusters

    np.allclose(np.linalg.norm(julearn_kmeans.cluster_centers_, axis=1), 1.0, atol=1e-5)

    # assert_array_equal(julearn_kmeans.labels_, sklearn_kmeans.labels_)

    # Verify that each point is closest to its assigned center
    for sample in X:
        distances = cosine_distances(julearn_kmeans.cluster_centers_, sample.reshape(1, -1)).flatten()
        assigned_center = np.argmin(distances)
        assert julearn_kmeans.labels_[np.where((X == sample).all(axis=1))[0][0]] == assigned_center


def test_kmeans_manhattan_metric():
    X = np.array([[1, 2, 1], [1, 4, 1], [1, 0, 1],
                  [4, 2, 1], [4, 4, 1], [4, 0, 1]], dtype=np.float32)
    random_state = 42
    n_clusters = 2

    print("Julearn KMeans cluster centers (manhattan metric)")
    julearn_kmeans = JulearnKMeans(n_clusters=n_clusters, random_state=random_state, metric="manhattan", verbose=199)
    julearn_kmeans.fit(X)

    assert julearn_kmeans.metric == "manhattan"
    assert len(julearn_kmeans.cluster_centers_) == n_clusters
    print("Cluster centers:\n", julearn_kmeans.cluster_centers_)
    print("Labels:\n", julearn_kmeans.labels_)
    print("X:\n", X)

    # Verify that each point is closest to its assigned center
    for sample in X:
        distances = manhattan_distances(julearn_kmeans.cluster_centers_, sample.reshape(1, -1)).flatten()
        assigned_center = np.argmin(distances)
        assert julearn_kmeans.labels_[np.where((X == sample).all(axis=1))[0][0]] == assigned_center
