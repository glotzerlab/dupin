import warnings

import numpy as np
import pytest
import ruptures as rpt

import dupin as du


class TestMeanShift:
    sensitivity = 1e-3
    min_mean_shift = 3.2907  # relative to std (use * 1.2 for safe tests)

    def get_shift_range(self, min_, max_):
        return (min_ * self.min_mean_shift, max_ * self.min_mean_shift)

    @pytest.fixture()
    def mean_shift(self):
        return du.preprocessing.filter.MeanShift(self.sensitivity)

    @pytest.fixture(params=range(3))
    def signal_without_mean_shift(self, seeds, request):
        i = request.param
        if i == 0:
            return rpt.pw_constant(
                100, 2, 0, 1, delta=self.get_shift_range(0.5, 0.8), seed=seeds()
            )[0]
        if i == 1:
            return rpt.pw_constant(
                100,
                2,
                1,
                2,
                delta=self.get_shift_range(-0.6, -0.5),
                seed=seeds(),
            )[0]
        return rpt.pw_constant(
            100, 2, 1, 1, delta=self.get_shift_range(0.5, 0.6), seed=seeds()
        )[0]

    def test_without_mean_shift(self, signal_without_mean_shift, mean_shift):
        filtered_signal = mean_shift(signal_without_mean_shift)
        assert filtered_signal.shape[1] == 0

    def test_error_on_small_signal(self, mean_shift):
        signal, _ = rpt.pw_constant(6, 1, 0, 2)
        with pytest.raises(
            ValueError,
            match="Signal to small to perform statistical analysis on.",
        ):
            mean_shift(signal)

    @pytest.fixture(params=range(4))
    def signal_with_mean_shift(self, seeds, request):
        i = request.param
        if i == 0:
            signal, _ = rpt.pw_constant(
                100,
                2,
                1,
                1,
                delta=self.get_shift_range(1.3, 1.35),
                seed=seeds(),
            )
            return 2, signal
        if i == 1:
            signal, _ = rpt.pw_constant(
                100,
                2,
                1,
                1,
                delta=self.get_shift_range(-1.4, -1.3),
                seed=seeds(),
            )
            return 2, signal
        if i == 2:  # noqa: PLR2004
            signal, _ = rpt.pw_constant(
                100,
                1,
                1,
                1,
                delta=self.get_shift_range(-1.35, -1.3),
                seed=seeds(),
            )
            no_signal, _ = rpt.pw_constant(100, 1, 0, 1, seed=seeds())
            return 1, np.concatenate((signal, no_signal), axis=1)

        signal, _ = rpt.pw_constant(
            100, 2, 1, 0.5, delta=self.get_shift_range(1.22, 1.25), seed=seeds()
        )
        no_signal, _ = rpt.pw_constant(100, 2, 0, 1, seed=seeds())
        return 2, np.concatenate((signal, no_signal), axis=1)

    def test_with_mean_shift(self, mean_shift, signal_with_mean_shift):
        n_dims, signal = signal_with_mean_shift
        filtered_signal = mean_shift(signal)
        assert filtered_signal.shape[1] == n_dims

    def test_return_filter(self, mean_shift, signal_with_mean_shift):
        n_dims, signal = signal_with_mean_shift
        filter_ = mean_shift(signal, return_filter=True)
        assert np.sum(filter_) == n_dims


class TestCorrelated:
    def generate_cov(self, rng, labels):
        N = len(labels)
        cov = np.zeros(shape=(N, N))
        for i in range(labels.max() + 1):
            index = np.flatnonzero(labels == i)
            x, y = np.meshgrid(index, index)
            cov[x, y] = rng.uniform(0.8, 1.0, size=x.size).reshape(x.shape)
        diag_indices = np.diag_indices_from(cov)
        cov[diag_indices] = 1.0
        flattened_cov = cov.ravel()
        zeros = np.flatnonzero(flattened_cov == 0)
        flattened_cov[zeros] = rng.uniform(0.2, 0.6, size=len(zeros))
        cov = np.triu(cov)
        return np.where(cov, cov, cov.T)

    def generate_labels(self, rng, n_clusters, n_dims=20):
        labels = np.zeros(n_dims, dtype=int)
        dims_per_cluster = n_dims // n_clusters
        available_dims = np.ones(n_dims, dtype=bool)
        dimensions = np.arange(n_dims, dtype=int)
        for i in range(1, n_clusters):
            new_labels = rng.choice(
                dimensions[available_dims], replace=False, size=dims_per_cluster
            )
            available_dims[new_labels] = False
            labels[new_labels] = i
        return labels

    @pytest.fixture(params=range(2, 5))
    def signal_and_cluster_labels(self, rng, request):
        n_clusters = request.param
        labels = self.generate_labels(rng, n_clusters)
        cov = self.generate_cov(rng, labels)
        # The covariance matrix is likely not positive semi-definite since the
        # correlations between x and y and z do not influece the correlation
        # between x and z in the generation method.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                rng.multivariate_normal(
                    mean=rng.uniform(-10, 10, size=20), cov=cov, size=50
                ),
                labels,
            )

    def check_equivalent_labels(self, a, b):
        map_ = {}
        for i, j in zip(a, b):
            map_[i] = j
        for i, j in map_.items():
            assert not np.any((a == i) != (b == j))

    @pytest.fixture()
    def correlated(self):
        return du.preprocessing.filter.Correlated(max_clusters=5)

    def test_with_set_clusters(self, signal_and_cluster_labels, correlated):
        signal, labels = signal_and_cluster_labels
        correlated(signal)
        assert correlated.n_clusters_ == np.unique(labels).shape[0]
        self.check_equivalent_labels(correlated.labels_, labels)

    @pytest.fixture(scope="module", params=(1, 2))
    def random_signal(self, rng):
        return rng.uniform(-10, 10, size=(10, 15))

    def test_zero_n_features(self, random_signal, correlated):
        with pytest.raises(
            ValueError, match="features_per_cluster must be 1 or greater."
        ):
            correlated(random_signal, features_per_cluster=0)

    def expected_n_features(self, n_features_per_cluster, correlated):
        """Get the expect number of features selected.

        Required to deal with random cluster number and sizes.
        """
        cluster_counts = np.unique(correlated.labels_, return_counts=True)[1]
        return min(
            len(correlated.labels_),
            np.minimum(n_features_per_cluster, cluster_counts).sum(),
        )

    @pytest.mark.parametrize("n_features", range(1, 4))
    def test_n_features(self, n_features, random_signal, correlated):
        filtered_signal = correlated(
            random_signal, features_per_cluster=n_features
        )
        expected_n_features = self.expected_n_features(n_features, correlated)
        assert filtered_signal.shape[1] == expected_n_features

    @pytest.mark.parametrize("n_features", range(1, 4))
    def test_return_filter(self, n_features, random_signal, correlated):
        filter_ = correlated(
            random_signal, return_filter=True, features_per_cluster=n_features
        )
        expected_n_features = self.expected_n_features(n_features, correlated)
        assert filter_.sum() == expected_n_features

    def test_feature_importance(self, random_signal, correlated):
        feature_importance = np.arange(15)
        filter_ = correlated(
            random_signal,
            features_per_cluster=2,
            return_filter=True,
            feature_importance=feature_importance,
        )
        for i in range(correlated.n_clusters_):
            features_in_cluster = np.flatnonzero(correlated.labels_ == i)
            assert np.all(filter_[features_in_cluster][-2:])
            assert not np.any(filter_[features_in_cluster][:-2])

    def test_invalid_construction(self):
        with pytest.raises(ValueError, match="Unsupported method aggregate."):
            du.preprocessing.filter.Correlated(method="aggregate")
        with pytest.raises(
            ValueError, match="Unsupported correlation option spearman."
        ):
            du.preprocessing.filter.Correlated(correlation="spearman")
        for n_clusters in (-1, 0, 1):
            with pytest.raises(
                ValueError, match="Max clusters must be greater than 1."
            ):
                du.preprocessing.filter.Correlated(max_clusters=n_clusters)
        for n_clusters in ("", {}):
            with pytest.raises(TypeError):
                du.preprocessing.filter.Correlated(max_clusters=n_clusters)

    def test_valid_construction(self, random_signal):
        max_clusters = 5
        correlated = du.preprocessing.filter.Correlated(
            max_clusters=max_clusters
        )
        assert correlated.max_clusters == max_clusters
        correlated(random_signal)
        # Expect scores for 2, 3, 4, and 5 clusters
        assert correlated.scores_.shape[0] == max_clusters - 1
        # These should error when called due to the faulty arguments passed
        # through.
        correlated = du.preprocessing.filter.Correlated(
            method_kwargs={"assign_labels": "foo"}
        )
        with pytest.raises(ValueError, match="foo"):
            correlated(random_signal)
        correlated = du.preprocessing.filter.Correlated(
            method_args=(5, {}, [12, 5])
        )
        with pytest.raises((TypeError, ValueError)):
            correlated(random_signal)
