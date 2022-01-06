"""Filters to reduce the dimensions of the signal."""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import bottleneck as bn
import numpy as np
import numpy.typing as npt
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

    def __init__(self, sensitivity: float):
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
    ) -> npt.ArrayLike:
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
    ) -> int:
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
                    RuntimeWarning,
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
    ) -> None:
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
    ) -> None:
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
        _logger.debug(f"Correlation: signal dimension, {signal.shape[1]}")
        if signal.shape[1] <= 2:
            self.labels_ = np.array([0, 1])
            self.scores_ = np.array([np.nan])
            self.n_clusters_ = 2
            if return_filter:
                return np.ones(2, dtype=bool)
            return np.copy(signal)

        sim_matrix, dist_matrix = self._get_similiarity_matrix(signal)
        isolated, connected = self._get_isolated(sim_matrix)
        sim_matrix, dist_matrix = self._handle_isolated(
            sim_matrix, dist_matrix, isolated, connected
        )

        self._cluster(sim_matrix, dist_matrix, connected)

        chosen_features = self._choose_features(
            feature_importance, features_per_cluster
        )

        filter_ = np.zeros(signal.shape[1], dtype=bool)
        filter_[np.flatnonzero(connected)[chosen_features]] = True
        self.filter_ = filter_

        if return_filter:
            return self.filter_
        return signal[:, self.filter_]

    def _get_similiarity_matrix(
        self, signal: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.correlation == "pearson":
            sim_matrix = np.abs(np.corrcoef(signal, rowvar=False))
            sim_matrix[np.isnan(sim_matrix)] = 0
            dist_matrix = np.abs(1 - sim_matrix)
            np.fill_diagonal(dist_matrix, 0)
            return sim_matrix, dist_matrix
        else:
            raise ValueError(
                f"Unsupported correlation type {self.correlation}."
            )

    def _get_isolated(self, sim_matrix: np.ndarray):
        connected = np.any(sim_matrix != 0, axis=1)
        return (np.logical_not(connected), connected)

    def _remove_isolated(
        self,
        sim_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        connected: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        else:
            raise ValueError(f"Unsupported method type {self.method}.")

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
    ) -> Tuple[np.ndarray, float]:
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_isolated = np.sum(isolated)
        if n_isolated >= 1:
            warnings.warn(
                f"{n_isolated} Isolated components detected " f"removing."
            )
            return self._remove_isolated(sim_matrix, dist_matrix, connected)
        return sim_matrix, dist_matrix

    def _cluster(
        self,
        sim_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        connected: np.ndarray,
    ) -> None:
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
    signal: :math:`(N_{samples}, N_{features})` np.ndarray of float
        The potentially multidimensional signal.
    dim: int, optional
        The dimension of spline to use, defaults to 1.
    spacing: int, optional
        The number of spaces beyond the dimension to space knots, defaults to
        ``None``. When ``None``, the behavior is :math:`\lceil d / 2 \rceil`.

    Returns
    -------
    feature_importance : :math:`(N_{features})` numpy.ndarray of float
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates that the fit was better.
    """
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
    likelihoods: :math:`(N_{features})` np.ndarray of float
        The likelihoods given from a `MeanShift` object or the likelihood that
        the given feature's signal happened by chance.

    Returns
    -------
    feature_importance : :math:`(N_{features})` numpy.ndarray of float
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates that the likelihood was lower.
    """
    return _to_unit_len(-likelihoods)


def noise_importance(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Rank features based on how standard deviation compares to the mean.

    Uses the rolling standard deviation over mean ignoring a mean of zero.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` np.ndarray of float
        The potentially multidimensional signal.
    window_size: int
        The size of rolling window to use.

    Returns
    -------
    feature_importance : :math:`(N_{features})` numpy.ndarray of float
        Feature rankings from 0 to 1 (higher is more important), for all
        features. A higher ranking indicates the standard deviation relative to
        the mean is low across the feature.
    """
    noise = np.nanmean(
        bn.move_std(signal, window_size, axis=0)
        / bn.move_mean(signal, window_size, axis=0),
        axis=0,
    )
    return _to_unit_len(-noise)
