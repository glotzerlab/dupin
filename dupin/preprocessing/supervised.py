"""Classes for use in utilizing supervised learning for event detection."""

from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np
import pandas as pd

from dupin import errors

try:
    import sklearn as sk
except ImportError:
    sk = errors._RaiseModuleError("sklearn")


def _str_isinstance(obj, classes):
    name = obj.__class__.__module__ + "." + obj.__class__.__name__
    for cls in classes:
        if name.endswith(cls):
            return True
    return False


def window_iter(seq: Sequence, window_size: int) -> Sequence:
    """Iterate over a sequence in slices of length window_size.

    Parameters
    ----------
    seq: list [``any``]
        The sequence to yield windows of.
    window_size: int
        The size of window iter iterator over.

    Yields
    ------
    window: list [ ``any`` ]
        The current window of the original data.
    """
    L = len(seq)
    for i, j in zip(range(0, L - window_size + 1), range(window_size, L + 1)):
        yield seq[i:j]


class Window:
    """Computes the error of a classifier discerning between halves of a window.

    The class implements a generic way of discerning the similiarity between
    nearby sections in a sequence through the use of a rolling window and
    machine learning classifiers. The class then outputs this similarity as a
    single dimension regardless of input size.

    The procedure is take a sliding window of a set size across the traectory.
    For each window, the left half is labeled as class 0 and the right as class
    one. The class then trains one or more weak classifiers for each window on a
    subset of points. The test loss on the remaining points is then
    aggregated across the classifiers and recorded. This testing loss is the
    single dimension representation of local signal similarity with higher
    values indicating dissimiliarity.

    Note:
        The returned signal will be smaller by ``window_size - 1`` than the
        original signal.

    Warning:
        For this to be useful, a *weak* classifier must be chosen. A weak
        classifier is one that has low discrimination ability. This prevents the
        training on noise between window halves. For small and intermediate
        window sizes, most classifiers will find noise that can (nearly)
        perfectly discriminate the halves of the window.

    Parameters
    ----------
    classifier : sklearn.base.ClassifierMixin
        A sklearn compatible classifier that is ready to fit to data.
    window_size : int
        The size of windows to learn on, should be a even number for best
        results.
    test_size : float
        Fraction of samples to use for computing the error through the loss
        function. This fraction is not fitted on.
    loss_function : ``callable`` [[`sklearn.base.ClassifierMixin`, \
                                `numpy.ndarray`, `numpy.ndarray`], \
                                `float`], optional
        A callable that takes in the fitted classifier, the test x and test y
        values and returns a loss (lower is better). By default this computes
        the zero-one loss if sklearn is available, otherwise this errors.
    store_intermediate_classifiers : `bool`, optional
        Whether to store the fitted classifier for each window in the sequence
        passed to `compute`. Defaults to False. **Warning**: If the classifier
        stores some or all of the sequence in fitting as is the case for
        kernelized classifiers, this optional will lead to significant
        increase in use of memory.
    n_classifiers : `int`, optional
        The number of classifiers and test train splits to use per window,
        defaults to 1. Higher numbers naturally smooth the error across a
        trajectory.
    combine_errors : `str`, optional
        What function to reduce the errors of ``n_classifiers`` with, defauts to
        "mean". Available values are "mean" and "median".
    """

    def __init__(
        self,
        classifier: "sk.base.ClassifierMixin",
        window_size: int,
        test_size: float,
        loss_function: Optional[
            Callable[["sk.base.ClassifierMixin", np.ndarray, np.ndarray], float]
        ] = None,
        store_intermediate_classifiers: bool = False,
        n_classifiers: int = 1,
        combine_errors: str = "mean",
    ) -> None:
        self.classifier = classifier
        self.window_size = window_size
        self.test_size = test_size
        if loss_function is None:
            loss_function = self._default_loss
        self.loss_function = loss_function
        self.store_intermediate_classifiers = store_intermediate_classifiers
        self.n_classifiers = n_classifiers
        self.combine_errors = combine_errors

    @property
    def window_size(self):
        """int: The size of windows to learn on."""
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        if value < 2:  # noqa: PLR2004
            msg = "window_size must be greater than 1."
            raise ValueError(msg)
        self._window_size = value

    @property
    def store_intermediate_classifiers(self):
        """bool: Whether to store the classifiers for each window.

        If ``True`` the classifiers are stored in ``classifiers_`` after calling
        ``compute``.
        """
        return self._store_intermediate_classifiers

    @store_intermediate_classifiers.setter
    def store_intermediate_classifiers(self, value):
        if not isinstance(value, bool):
            msg = "Expected bool for store_intermediate_classifiers."
            raise TypeError(msg)
        self._store_intermediate_classifiers = value

    @property
    def loss_function(self):
        """``callable`` [[ `sklearn.base.ClassifierMixin`, `numpy.ndarray`, \
                `numpy.ndarray` ], `float` ]: Returns the loss for a fitted \
        classifier given the test x and y.
        """  # noqa: D205
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        if not callable(value):
            msg = "loss_function must be callable."
            raise TypeError(msg)
        self._loss_function = value

    @property
    def test_size(self):
        """float: Fraction of samples to use for computing the error."""
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        if value <= 0.0 or value >= 1.0:  # noqa: PLR2004
            msg = "test_size must be between 0 and 1."
            raise ValueError(msg)
        self._test_size = value

    @property
    def n_classifiers(self):
        """int: Number of classifiers and test-train splits per window.

        Higher numbers naturally smooth the error across a trajectory.
        """
        return self._n_classifiers

    @n_classifiers.setter
    def n_classifiers(self, value):
        if value < 1:
            msg = "n_classifiers must be greater than 0."
            raise ValueError(msg)
        self._n_classifiers = value

    @property
    def combine_errors(self):
        """str: What function to reduce the errors of ``n_classifiers`` with.

        Available values are "mean" and "median".
        """
        return self._combine_errors

    @combine_errors.setter
    def combine_errors(self, value):
        if value not in ("mean", "median"):
            msg = "combine_errors must be in ('mean', 'median')."
            raise ValueError(msg)
        self._combine_errors = value

    @property
    def _reduce(self):
        return np.mean if self.combine_errors == "mean" else np.median

    def compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the loss for classifiers trained on discerning window halves.

        Parameters
        ----------
        X : (:math:`T`, :math:`N_f`) np.ndarray
            An NumPy array where the first dimension is time or sequence
            progression and the second is features.

        Returns
        -------
        errors : list
            Returns the list of loss function values for each window in ``X``.
        """
        if isinstance(X, pd.core.frame.DataFrame):
            return self.compute(X.to_numpy())

        errors = []
        if self.store_intermediate_classifiers:
            self.classifiers_ = []
        y = np.repeat([0, 1], np.ceil(self.window_size / 2))[: self.window_size]
        shuffle_splits = sk.model_selection.StratifiedShuffleSplit(
            n_splits=self.n_classifiers, test_size=self.test_size
        )
        for x in window_iter(X, self.window_size):
            if self.store_intermediate_classifiers:
                self.classifiers_.append([])
            slice_errors = []
            for train_indices, test_indices in shuffle_splits.split(x, y):
                self.classifier.fit(x[train_indices], y[train_indices])
                slice_errors.append(
                    self._loss_function(
                        self.classifier,
                        x[test_indices],
                        y[test_indices],
                    )
                )
                # If storing intermediate classifiers clone the classifier to
                # ensure we train/fit on a new identical model.
                if self.store_intermediate_classifiers:
                    self.classifiers_[-1].append(self.classifier)
                    self.classifier = sk.base.clone(self.classifier)
            errors.append(self._reduce(slice_errors))
        self.errors = np.array(errors)
        return self.errors

    @staticmethod
    def _default_loss(classifier, x, y):
        return sk.metrics.zero_one_loss(y, classifier.predict(x))
