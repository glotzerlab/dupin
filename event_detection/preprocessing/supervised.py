"""Classes for use in utilizing supervised learning for event detection."""

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

import event_detection.errors as errors

try:
    import sklearn as sk
except ImportError:
    sk = errors._RaiseModuleError("sklearn")


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
    nearby sections in a sequence through the use of a rolling window. If the
    sequence is changing across its length then the classifier should do a very
    good job of distinguishing between window halves. If the sequence is static,
    then the classifier should approach some error up to an error rate of 0.5.

    Warning:
        For this to be useful, a *weak* classifier must be chosen. A weak
        classifier is one that has low discrimination ability. This prevents the
        training on noise between window halves. For small and intermediate
        window sizes, most classifiers will find noise that can (nearly)
        perfectly discriminate the halves of the window.
    """

    def __init__(
        self,
        classifier: "sk.base.ClassifierMixin",
        window_size: int,
        test_size: float,
        loss_function: Optional[
            Callable[[np.ndarray, np.ndarray], float]
        ] = None,
        store_intermediate_classifiers: bool = False,
        n_classifiers: int = 1,
        combine_errors: str = "mean",
    ) -> None:
        """Create a `Window` object.

        Parameters
        ----------
        classifier : sklearn compatible classifier
            A sklearn compatible classifier that is ready to fit to data.
        window_size : int
            The size of windows to learn on, should be a even number for best
            results.
        test_size : float
            Fraction of samples to use for computing the error through the loss
            function. This fraction is not fitted on.
        loss_function : callable[[sklearn.base.ClassifierMixin, \
                                  np.ndarray, np.ndarray], float], optional
            A callable that takes in the fitted classifier, the test x and test
            y values and returns a loss (lower is better). By default this
            computes the zero-one loss if `sklearn` is available, otherwise this
            errors.
        store_intermediate_classifiers : bool, optional
            Whether to store the fitted classifier for each window in the
            sequence passed to `compute`. Defaults to False. **Warning**: If the
            classifier stores some or all of the sequence in fitting as is the
            case for kernelized classifiers, this optional will lead a much use
            of memory.
        n_classifiers : int, optional
            The number of classifiers and test train splits to use per window,
            defaults to 1. Higher numbers naturally smooth the error across a
            trajectory.
        combine_errors : str, optional
            What function to reduce the errors of ``n_classifiers`` which,
            defauts to "mean". Available values are "mean" and "median".
        """
        self._classifier = classifier
        self.window_size = window_size
        self.test_size = test_size
        if loss_function is None:
            self._loss_function = self._default_loss
        else:
            self._loss_function = loss_function
        self.store_intermediate_classifiers = store_intermediate_classifiers
        self.n_classifiers = n_classifiers
        self.combine_errors = combine_errors

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
        errors = []
        if self.store_intermediate_classifiers:
            self._classifiers = []
        y = np.repeat([0, 1], np.ceil(self.window_size / 2))[: self.window_size]
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        shuffle_splits = sk.model_selection.StratifiedShuffleSplit(
            n_splits=self.n_classifiers, test_size=self.test_size
        )
        for x in window_iter(X, self.window_size):
            if self.store_intermediate_classifiers:
                self._classifiers.append([])
            slice_errors = []
            for train_indices, test_indices in shuffle_splits.split(x, y):
                self._classifier.fit(x[train_indices], y[train_indices])
                slice_errors.append(
                    self._loss_function(
                        self._classifier,
                        x[test_indices],
                        y[test_indices],
                    )
                )
                # If storing intermediate classifiers clone the classifier to
                # ensure we train/fit on a new identical model.
                if self.store_intermediate_classifiers:
                    self._classifiers[-1].append(self._classifier)
                    self._classifier = sk.base.clone(self._classifier)
            errors.append(self._reduce(slice_errors))
        self.errors = np.array(errors)
        return self.errors

    @staticmethod
    def _default_loss(classifier, x, y):
        return sk.metrics.zero_one_loss(y, classifier(x))
