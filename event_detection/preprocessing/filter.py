"""Filters to reduce the dimensions of the signal."""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp
import sklearn as sk
import sklearn.cluster
import sklearn.metrics

_logger = logging.getLogger()


class MeanShift:
    """Filter out dimensions that don't undergo a significant shift in mean.

    The filter computes the mean and standard deviation of both ends of the
    signal, and determines whether the mean of one either end is statistically
    significant (judged by ``sensitivity``) compared to the other. The filter
    assumes Gaussian noise.

    Attributes
    ----------
    mean_shifts_: :math:`(N_{features},) np.ndarray of float
        The maximum number of standard deviations between the means of the two
        ends of the last computed signal.
    likelihoods_: :math:`(N_{features},) np.ndarray of float
        The likelihood that such a mean shift would be observed in a Gaussian
        with the given mean and standard deviation. The likelihood discounts
        signal length.
    filter_: :math:`(N_{features})` numpy.ndarray of bool
        The array of features selected.
    """

    def __init__(self, sensitivity):
        """Create a MeanShift filter.

        Parameters
        ----------
        sensitivity: float, optional
            The minimum likelihood that one of the signal's end's mean is drawn
            from the Gaussian approximation of the other end to require. In
            other words, the lower the number the increased probability that the
            difference in means is not random. Defaults to 0.01.
        """
        self.sensitivity = sensitivity

    def __call__(
        self,
        signal: npt.ArrayLike,
        sample_size: Union[int, float] = 0.1,
        return_filter: Optional[bool] = False,
    ):
        """Filter dimensions without a detected mean shift.

        Parameters
        ----------
        signal : :math:`(N_{samples}, N_{features})` numpy.ndarray of float
            The signal to filter dimensions from.
        sample_size: float or int, optional
            Either the fraction of the overall signal to use to evaluate the
            statistics of each end of the signal, or the number of data points
            to use on each end of the signal for statistics. Default to 0.1. If
            this would result in less than three data points, three will be
            used.
        return_filter: bool, optional
            Whether to return the Boolean array filter rather than the filtered
            data. Defaults to ``False``.

        Returns
        -------
        :math:`(N_{samples}, N_{filtered})` numpy.ndarray of float or \
                :math:`(N_{features})` numpy.ndarray of bool
            By default returns the filtered data with features deemed
            insignificant removed. If ``return_filter`` is ``True``, the Boolean
            array filtering features is returned.
        """
        n_frames = self._get_sample_size(len(signal), sample_size)
        start, end = signal[:n_frames], signal[-n_frames:]

        self.mean_shifts_ = self._get_mean_shift_std(start, end)
        self.likelihoods_ = self._get_likelihood(self.mean_shifts_)

        self.filter_ = self.likelihoods_ <= self.sensitivity
        if return_filter:
            return self.filter_
        return signal[:, self.filter_]

    @staticmethod
    def _get_sample_size(
        signal_len: int, sample_size: Optional[Union[int, float]]
    ):
        # 3 on each side is the minimum to perform this analysis.
        if signal_len < 7:
            raise ValueError(
                "Signal to small to perform statistical analysis on."
            )
        if isinstance(sample_size, float):
            sample_size = max(3, int(sample_size * signal_len))

        if sample_size >= signal_len / 2:
            raise ValueError(
                f"Cannot use {sample_size} frames with a signal of "
                f"length {signal_len}."
            )
        return sample_size

    @staticmethod
    def _get_mean_shift_std(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu_a, std_a = a.mean(axis=0), a.std(axis=0)
        mu_b, std_b = b.mean(axis=0), b.std(axis=0)
        shift_ab = np.abs(mu_b - mu_a) / std_a
        shift_ba = np.abs(mu_a - mu_b) / std_b
        return np.maximum(shift_ab, shift_ba)

    @staticmethod
    def _get_likelihood(std_shift: np.ndarray) -> np.ndarray:
        return 1 - sp.special.erf(std_shift / np.sqrt(2))


class Correlated:
    """Filter out dimensions that are highly correlated with each other.

    The filter computes the specified correlation matrix, and clusters the
    features based on the distance or similarity matrix depending on the
    specified clustering method. The number of clusters is determined by the
    minimum avaerage silhouette score for each number of clusters tested. Then a
    set number of features from each cluster is chosen through provided feature
    importance or randomly.

    Attributes
    ----------
    method: str
        The method to use. Current options are "spectral".
    correlation: str
        The correlation type to use for computing similarity and distance
        matrices. Currently supported options are "pearson".
    max_clusters: int
        The maximum number of clusters to try. Defaults to 10.
    n_clusters_: int
        The determined optimal number of clusters.
    labels_: :math:`(N_{features},) numpy.ndarray of int
        The cluster labels for the best performing number of clusters.
    scores_: :math:`(N - 2,)` numpy.ndarray of float
        The scores for each number of clusters tried. Starts at 2.
    filter_: :math:`(N_{features})` numpy.ndarray of bool
        The array of features selected.
    """

    def __init__(
        self,
        method: str = "spectral",
        correlation: str = "pearson",
        max_clusters: int = 10,
        method_args: Tuple[Any, ...] = (),
        method_kwargs: Dict[str, Any] = None,
    ):
        """Construct a Correlated filter.

        Parameters
        ----------
        method: str, optional
            The method to use. Current options are "spectral". Defaults to
            "spectral".
        correlation: str, optional
            The correlation type to use for computing similarity and distance
            matrices. Currently supported options are "pearson". Defaults to
            "pearson".
        max_clusters: int, optional
            The maximum number of clusters to try. Defaults to 10.
        method_args: tuple, optional
            Any positional arguments to pass to the selected method's
            construction.
        method_kwargs: dict[str, ``any``], optional
            Any keyword arguments to pass to the selected method's construction.
        """
        self.method = method
        self.correlation = correlation
        self.max_clusters = max_clusters
        self._method_args = method_args
        self._method_kwargs = {} if method_kwargs is None else method_kwargs

    def __call__(
        self,
        signal: npt.ArrayLike,
        features_per_cluster: int = 1,
        return_filter: bool = False,
        feature_importance: Optional[npt.ArrayLike] = None,
    ):
        """Filter out correlated features.

        Parameters
        ----------
        signal : :math:`(N_{samples}, N_{features})` numpy.ndarray of float
            The signal to filter dimensions from.
        features_per_cluster: int, optional
            The number of features to keep per cluster. Defaults to 1.
        return_filter: bool, optimal
            Whether to return the features selected or not. Defaults to False.
        feature_importance: :math:`(N_{features},)` numpy.ndarray of float, \
                optional
            The importances of each feature. This determines which feature(s)
            from each cluster are chosen. If not provided, random importances
            are used.

        Returns
        -------
        :math:`(N_{samples}, N_{filtered})` numpy.ndarray of float or \
                :math:`(N_{features})` numpy.ndarray of bool
            By default returns the filtered data with features deemed
            insignificant removed. If ``return_filter`` is ``True``, the Boolean
            array filtering features is returned.
        """
        similarity_matrix, distance_matrix = self._get_similiarity_matrix(
            signal
        )
        if feature_importance is None:
            rng = np.random.default_rng()
            feature_importance = rng.random(signal.shape[1])

        cluster_ids = []
        scores = []
        for n_clusters in range(2, self.max_clusters + 1):
            labels, score = self._compute_clusters(
                n_clusters, similarity_matrix, distance_matrix
            )
            cluster_ids.append(labels)
            scores.append(score)

        self.scores_ = np.array(scores)
        best_cluster_index = self.scores_.argmin()
        self.labels_ = cluster_ids[best_cluster_index]
        self.n_clusters_ = best_cluster_index + 2

        filter_ = np.zeros(similarity_matrix.shape[0], dtype=bool)
        filter_[
            self._choose_features(feature_importance, features_per_cluster)
        ] = True
        self.filter_ = filter_

        if return_filter:
            return self.filter_
        return signal[:, self.filter_]

    def _get_similiarity_matrix(
        self, signal: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.correlation == "pearson":
            similarity_matrix = np.abs(np.corrcoef(signal, rowvar=False))
            distance_matrix = np.abs(1 - similarity_matrix)
            return similarity_matrix, distance_matrix
        else:
            raise ValueError(
                f"Unsupported correlation type {self.correlation}."
            )

    def _get_method_instance(self, n_clusters: int) -> sk.base.ClusterMixin:
        if self.method == "spectral":
            return sk.cluster.SpectralClustering(
                n_clusters,
                *self._method_args,
                affinity="precomputed",
                **self._method_kwargs,
            )
        else:
            raise ValueError(f"Unsupported method type {self.method}.")

    def _choose_features(
        self, feature_importance: np.ndarray, features_per_cluster: int
    ) -> np.ndarray:
        ids = self.labels_
        features = []
        for label in range(self.n_clusters_):
            in_cluster = ids == label
            sorted_indices = np.argsort(feature_importance[in_cluster])
            features.append(
                np.flatnonzero(in_cluster)[sorted_indices][
                    :features_per_cluster
                ]
            )
        return np.concatenate(features)

    def _compute_clusters(
        self,
        n_clusters: int,
        similarity_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        clusterer = self._get_method_instance(n_clusters)
        clusterer.fit(similarity_matrix)
        score = sk.metrics.silhouette_score(
            distance_matrix,
            metric="precomputed",
            labels=clusterer.labels_,
        )
        return clusterer.labels_, score
