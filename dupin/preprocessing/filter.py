"""Feature selection schemes.

This provides feature selection schemes distinct from packages like
scikit-learn. These packages can easily be used as well for feature selection.
"""

import logging
import warnings
from typing import Any, Optional, Union

import bottleneck as bn
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import sklearn as sk
import sklearn.cluster
import sklearn.metrics

_logger = logging.getLogger(__name__)


class MeanShift:
    """Filter out dimensions that don't undergo a significant shift in mean.

    The filter computes the mean and standard deviation of both ends of the
    signal, and determines whether the mean of one either end is statistically
    significant (judged by ``sensitivity``) compared to the other. The filter
    assumes Gaussian noise.

    Parameters
    ----------
    sensitivity: `float`, optional
        The minimum likelihood that one of the signal's end's mean is drawn
        from the Gaussian approximation of the other end to require. In
        other words, the lower the number the increased probability that the
        difference in means is not random. Defaults to 0.01.

    Attributes
    ----------
    sensitivity: float
        The minimum likelihood that one of the signal's end's mean is drawn
        from the Gaussian approximation of the other end to require. In
        other words, the lower the number the increased probability that the
        difference in means is not random.
    mean_shifts_: :math:`(N_{features},)` `numpy.ndarray` of `float`
        The maximum number of standard deviations between the means of the two
        ends of the last computed signal.
    likelihoods_: :math:`(N_{features},)` `numpy.ndarray` of `float`
        The likelihood that such a mean shift would be observed in a Gaussian
        with the given mean and standard deviation. The likelihood discounts
        signal length.
    filter_: :math:`(N_{features})` `numpy.ndarray` of `bool`
        The array of features selected.
    """

    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity

    def __call__(
        self,
        signal: npt.ArrayLike,
        sample_size: Union[int, float] = 0.1,
        return_filter: Optional[bool] = False,
    ) -> npt.ArrayLike:
        """Filter dimensions without a detected mean shift.

        Parameters
        ----------
        signal : :math:`(N_{samples}, N_{features})` `numpy.ndarray` of `float`
            The signal to filter dimensions from.
        sample_size: `float` or `int`, optional
            Either the fraction of the overall signal to use to evaluate the
            statistics of each end of the signal, or the number of data points
            to use on each end of the signal for statistics. Default to 0.1. If
            this would result in less than three data points, three will be
            used.
        return_filter: `bool`, optional
            Whether to return the Boolean array filter rather than the filtered
            data. Defaults to ``False``.

        Returns
        -------
        :math:`(N_{samples}, N_{filtered})` `numpy.ndarray` of `float` or \
                :math:`(N_{features})` `numpy.ndarray` of `bool`
            By default returns the filtered data with features deemed
            insignificant removed. If ``return_filter`` is ``True``, the Boolean
            array filtering features is returned.
        """
        if isinstance(signal, pd.DataFrame) and not return_filter:
            filter_ = self(signal.to_numpy(), sample_size, True)
            return signal.iloc[:, filter_]

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
    ) -> int:
        # 3 on each side is the minimum to perform this analysis.
        min_signal_length = 7
        if signal_len < min_signal_length:
            msg = "Signal to small to perform statistical analysis on."
            raise ValueError(msg)
        if isinstance(sample_size, float):
            sample_size = max(3, int(sample_size * signal_len))

        if sample_size >= signal_len / 2:
            msg = (
                f"Cannot use {sample_size} frames with a signal of "
                f"length {signal_len}."
            )
            raise ValueError(msg)
        return sample_size

    @staticmethod
    def _get_mean_shift_std(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu_a, std_a = a.mean(axis=0), a.std(axis=0)
        mu_b, std_b = b.mean(axis=0), b.std(axis=0)

        def calc_mean_shift_with_warn(diff, std):
            zeros_found = False
            with warnings.catch_warnings(record=True) as warns:
                shift = diff / std
                if len(warns) != 0:
                    zeros_found = True
            if zeros_found:
                mask = std == 0
                diff_zeros = diff == 0
                shift[mask & np.logical_not(diff_zeros)] = np.infty
                shift[mask & diff_zeros] = 0
                warnings.warn(
                    "MeanShift: Zero standard deviation found.",
                    stacklevel=2,
                )
            return shift

        diff = np.abs(mu_a - mu_b)
        shift_ab = calc_mean_shift_with_warn(diff, std_a)
        shift_ba = calc_mean_shift_with_warn(diff, std_b)
        return np.maximum(shift_ab, shift_ba)

    @staticmethod
    def _get_likelihood(std_shift: np.ndarray) -> np.ndarray:
        return 1 - sp.special.erf(std_shift / np.sqrt(2))


class Correlated:
    """Filter out dimensions that are highly correlated with each other.

    The filter computes the chosen feature correlation matrix, and clusters the
    features based on the distance or similarity matrix depending on the
    specified clustering method. The number of clusters is determined by the
    minimum avaerage silhouette score for each number of clusters tested. Then a
    set number of features from each cluster is chosen through provided feature
    importance or randomly.

    Parameters
    ----------
    method: `str`, optional
        The method to use. Current options are "spectral". Defaults to
        "spectral".
    correlation: `str`, optional
        The correlation type to use for computing similarity and distance
        matrices. Currently supported options are "pearson". Defaults to
        "pearson".
    max_clusters: `int`, optional
        The maximum number of clusters to try. Defaults to 10.
    method_args: `tuple`, optional
        Any positional arguments to pass to the selected method's
        construction.
    method_kwargs: `dict` [`str`, ``any`` ], optional
        Any keyword arguments to pass to the selected method's construction.

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
    labels_: :math:`(N_{features},)` `numpy.ndarray` of `int`
        The cluster labels for the best performing number of clusters.
    scores_: :math:`(N - 2,)` `numpy.ndarray` of `float`
        The scores for each number of clusters tried. Starts at 2.
    filter_: :math:`(N_{features})` `numpy.ndarray` of `bool`
        The array of features selected.
    """

    _methods = frozenset(("spectral",))
    _correlations = frozenset(("pearson",))

    def __init__(
        self,
        method: str = "spectral",
        correlation: str = "pearson",
        max_clusters: int = 10,
        method_args: tuple[Any, ...] = (),
        method_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if method not in self._methods:
            msg = (
                f"Unsupported method {method}. Supported options "
                f"{self._methods}"
            )
            raise ValueError(msg)
        self.method = method
        if correlation not in self._correlations:
            msg = (
                f"Unsupported correlation option {correlation}. Supported "
                f"options {self._correlations}"
            )
            raise ValueError(msg)
        self.correlation = correlation
        min_clusters = 2
        if max_clusters < min_clusters:
            msg = "Max clusters must be greater than 1."
            raise ValueError(msg)
        self.max_clusters = max_clusters
        self._method_args = method_args
        self._method_kwargs = {} if method_kwargs is None else method_kwargs

    def __call__(
        self,
        signal: npt.ArrayLike,
        features_per_cluster: int = 1,
        return_filter: bool = False,
        feature_importance: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Filter out correlated features.

        Parameters
        ----------
        signal : :math:`(N_{samples}, N_{features})` `numpy.ndarray` of `float`
            The signal to filter dimensions from.
        features_per_cluster: `int`, optional
            The number of features to keep per cluster. Defaults to 1.
        return_filter: `bool`, optional
            Whether to return the features selected or not. Defaults to False.
        feature_importance: :math:`(N_{features},)` `numpy.ndarray` of `float`\
                , optional
            The importances of each feature. This determines which feature(s)
            from each cluster are chosen. If not provided, random importances
            are used.

        Returns
        -------
        :math:`(N_{samples}, N_{filtered})` `numpy.ndarray` of `float` or \
                :math:`(N_{features})` `numpy.ndarray` of `bool`
            By default returns the filtered data with features deemed
            insignificant removed. If ``return_filter`` is ``True``, the Boolean
            array filtering features is returned.
        """
        s = signal.to_numpy() if isinstance(signal, pd.DataFrame) else signal
        feature_mask = self._get_feature_mask(
            s, features_per_cluster, feature_importance
        )
        self.filter_ = feature_mask
        if return_filter:
            return feature_mask
        if isinstance(signal, pd.DataFrame):
            return signal.iloc[:, feature_mask]
        return signal[:, feature_mask]

    def _get_feature_mask(
        self,
        signal: npt.ArrayLike,
        features_per_cluster: int = 1,
        feature_importance: Optional[npt.ArrayLike] = None,
    ) -> np.ndarray:
        _logger.debug(f"Correlation: signal dimension, {signal.shape[1]}")
        if features_per_cluster < 1:
            msg = "features_per_cluster must be 1 or greater."
            raise ValueError(msg)

        connected_features = self._cluster(signal)
        chosen_features = self._choose_features(
            feature_importance, features_per_cluster
        )

        filter_ = np.zeros(signal.shape[1], dtype=bool)
        filter_[np.flatnonzero(connected_features)[chosen_features]] = True
        return filter_

    def _get_similiarity_matrix(
        self, signal: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.correlation == "pearson":
            sim_matrix = np.abs(np.corrcoef(signal, rowvar=False))
            sim_matrix[np.isnan(sim_matrix)] = 0
            dist_matrix = np.abs(1 - sim_matrix)
            np.fill_diagonal(dist_matrix, 0)
            return sim_matrix, dist_matrix
        msg = f"Unsupported correlation type {self.correlation}."
        raise ValueError(msg)

    def _get_isolated(self, sim_matrix: np.ndarray):
        connected = np.any(sim_matrix != 0, axis=1)
        return (np.logical_not(connected), connected)

    def _remove_isolated(
        self,
        sim_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        connected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            sim_matrix[:, connected][connected, :],
            dist_matrix[:, connected][connected, :],
        )

    def _get_method_instance(self, n_clusters: int) -> sk.base.ClusterMixin:
        if self.method == "spectral":
            return sk.cluster.SpectralClustering(
                n_clusters,
                *self._method_args,
                affinity="precomputed",
                **self._method_kwargs,
            )
        msg = f"Unsupported method type {self.method}."
        raise ValueError(msg)

    def _choose_features(
        self, feature_importance: np.ndarray, features_per_cluster: int
    ) -> np.ndarray:
        if feature_importance is None:
            rng = np.random.default_rng()
            feature_importance = rng.random(len(self.labels_))

        ids = self.labels_
        features = []
        for label in range(self.n_clusters_):
            in_cluster = ids == label
            sorted_indices = np.argsort(-feature_importance[in_cluster])
            features.append(
                np.flatnonzero(in_cluster)[sorted_indices][
                    :features_per_cluster
                ]
            )
        return np.concatenate(features)

    def _compute_clusters(
        self,
        n_clusters: int,
        sim_matrix: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        clusterer = self._get_method_instance(n_clusters)
        clusterer.fit(sim_matrix)
        score = sk.metrics.silhouette_score(
            dist_matrix,
            metric="precomputed",
            labels=clusterer.labels_,
        )
        return clusterer.labels_, score

    def _handle_isolated(
        self,
        sim_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        isolated: np.ndarray,
        connected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_isolated = np.sum(isolated)
        if n_isolated >= 1:
            warnings.warn(
                f"{n_isolated} Isolated components detected removing.",
                stacklevel=2,
            )
            return self._remove_isolated(sim_matrix, dist_matrix, connected)
        return sim_matrix, dist_matrix

    def _cluster(self, signal: np.ndarray) -> None:
        if signal.shape[1] < 3:  # noqa: PLR2004
            self._low_dimension_clustering(signal.shape[1])
            return None

        sim_matrix, dist_matrix = self._get_similiarity_matrix(signal)
        isolated, connected = self._get_isolated(sim_matrix)
        sim_matrix, dist_matrix = self._handle_isolated(
            sim_matrix, dist_matrix, isolated, connected
        )
        cluster_ids = []
        scores = []
        for n_clusters in range(2, min(len(sim_matrix), self.max_clusters + 1)):
            labels, score = self._compute_clusters(
                n_clusters, sim_matrix, dist_matrix
            )
            cluster_ids.append(labels)
            scores.append(score)

        self.scores_ = np.array(scores)
        best_cluster_index = self.scores_.argmax()
        self.labels_ = np.full(connected.shape[0], -1, dtype=int)
        self.labels_[connected] = cluster_ids[best_cluster_index]
        self.n_clusters_ = best_cluster_index + 2
        return connected

    def _low_dimension_clustering(self, size: int) -> None:
        if size == 1:
            self.scores_ = np.array([np.nan])
            self.labels_ = np.zeros(1)
            self.n_clusters_ = 1
            return
        # Size must be 2
        self.scores_ = np.array([np.nan])
        self.labels_ = np.array([0, 1])
        self.n_clusters_ = 2


def _to_unit_len(arr):
    min_, max_ = arr.min(), arr.max()
    if min_ == max_:
        return np.zeros_like(arr)
    return (arr - min_) / (max_ - min_)


def local_smoothness_importance(
    signal: np.ndarray, dim: int = 1, spacing: Optional[int] = None
) -> np.ndarray:
    r"""Rank features based on how well a spaced LSQ spline fits the feature.

    Uses the negative MSE projected to a range of :math:`[0, 1]`.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` `numpy.ndarray` of `float`
        The potentially multidimensional signal.
    dim: `int`, optional
        The dimension of spline to use, defaults to 1.
    spacing: `int`, optional
        The number of spaces beyond the dimension to space knots, defaults to
        ``None``. When ``None``, the behavior is :math:`\lceil d / 2 \rceil`.

    Returns
    -------
    feature_importance : :math:`(N_{features})` `numpy.ndarray` of `float`
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates that the fit was better.
    """
    if isinstance(signal, pd.DataFrame):
        return local_smoothness_importance(signal.to_numpy(), dim, spacing)
    x = np.arange(signal.shape[0])
    spacing = dim + int(np.ceil(dim / 2)) if spacing is None else dim + spacing
    beg = dim + 2
    start, end = (x[0],) * (dim + 1), (x[-1],) * (dim + 1)
    knots = np.r_[start, x[beg:-beg:spacing], end]
    spline = sp.interpolate.make_lsq_spline(x, signal, t=knots, k=dim)
    mse = np.sqrt(np.sum((signal - spline(x)) ** 2, axis=0))
    return _to_unit_len(-mse)


def mean_shift_importance(likelihoods: np.ndarray) -> np.ndarray:
    """Rank features based on how strong of a mean shift they have.

    Parameters
    ----------
    likelihoods: :math:`(N_{features})` `numpy.ndarray` of `float`
        The likelihoods given from a `MeanShift` object or the likelihood that
        the given feature's signal happened by chance.

    Returns
    -------
    feature_importance : :math:`(N_{features})` `numpy.ndarray` of `float`
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates that the likelihood was lower.
    """
    if isinstance(likelihoods, pd.DataFrame):
        return mean_shift_importance(likelihoods.to_numpy)
    return _to_unit_len(-likelihoods)


def jump_size_importance(signal: np.ndarray, n_end: int = 3) -> np.ndarray:
    """Rank features based on the size of the relative difference between ends.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` `numpy.ndarray` of `float`
        The potentially multidimensional signal.
    n_end: `int`, optional
        The number of indices to take on either end to compute the mean to
        determine the jump from one end to the other.

    Returns
    -------
    feature_importance : :math:`(N_{features})` `numpy.ndarray` of `float`
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates the relative magnitude of the jump
        or drop between signal ends is larger.
    """
    if isinstance(signal, pd.DataFrame):
        return jump_size_importance(signal.to_numpy(), n_end)
    left = signal[:n_end].mean(axis=0)
    right = signal[-n_end:].mean(axis=0)
    jump = (np.maximum(left, right) - np.minimum(left, right)) / np.minimum(
        left, right
    )
    return _to_unit_len(-jump)


def noise_importance(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Rank features based on how standard deviation compares to the mean.

    Uses the rolling standard deviation over mean ignoring a mean of zero.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` `numpy.ndarray` of `float`
        The potentially multidimensional signal.
    window_size: int
        The size of rolling window to use.

    Returns
    -------
    feature_importance : :math:`(N_{features})` `numpy.ndarray` of `float`
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates the standard deviation relative to
        the mean is low across the feature.
    """
    if isinstance(signal, pd.DataFrame):
        return noise_importance(signal.to_numpy(), window_size)
    noise = np.nanmean(
        bn.move_std(signal, window_size, axis=0)
        / bn.move_mean(signal, window_size, axis=0),
        axis=0,
    )
    return _to_unit_len(-noise)
