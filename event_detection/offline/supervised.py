"""Classes for use in utilizing supervised learning for event detection."""
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import sklearn as sk


def window_iter(seq, window_size):
    """Iterate over a sequence in slices of length window_size."""
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
        For this to be useful, a *weak* classifier must be chosen. By weak, a
        classifier that has low discrimination ability is meant. This prevents
        the training on noise between window halves. For small and intermediate
        window sizes, most classifiers will find noise that can (nearly)
        perfectly discriminate the halves of the window.
    """

    def __init__(
        self,
        classifier,
        window_size: int,
        test_size: float,
        loss_function: Optional[
            Callable[[np.ndarray, np.ndarray], float]
        ] = None,
        store_intermediate_classifiers: bool = False,
    ):
        """Create a `Window` object.

        Parameters
        ----------
        classifier : sklearn compatible classifier
            A sklearn compatible classifier that is ready to fit to data. Should
            be a weak classifier (see class documentation warning).
        window_size : int
            The size of windows to learn on, should be a even number for best
            results.
        test_size : float
            Fraction of samples to use for computing the error through the loss
            function. This fraction is not fitted on.
        loss_function : callable[[hp.ndarray, np.ndarray], float], optional
            A callable that takes in the predicted and actual class labels for a
            given window and outputs a score. Examples of this include the
            zero-one loss and logistic loss. Defaults to the zero-one loss.
        store_intermediate_classifiers : bool, optional
            Whether to store the fitted classifier for each window in the
            sequence passed to `compute`. Defaults to False. **Warning**: If the
            classifier stores some or all of the sequence in fitting as is the
            case for kernelized memory this could lead to a drastic increase of
            memory for this class.
        """
        self._classifier = classifier
        self.window_size = window_size
        self.test_size = test_size
        if loss_function is None:
            self._loss_function = sk.metrics.zero_one_loss
        else:
            self._loss_function = loss_function
        self.store_intermediate_classifiers = store_intermediate_classifiers

    def compute(self, X: np.ndarray) -> List[float]:
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
        self.errors = []
        if self.store_intermediate_classifiers:
            self._classifiers = []
        y = np.repeat([0, 1], np.ceil(self.window_size / 2))[: self.window_size]
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        for x in window_iter(X, self.window_size):
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = sk.model_selection.train_test_split(
                x,
                y,
                test_size=self.test_size,
                stratify=y,
            )
            self._classifier.fit(x_train, y_train)
            self.errors.append(
                self._loss_function(y_test, self._classifier.predict(x_test))
            )
            # If storing intermediate classifiers clone the classifier to ensure
            # we train/fit on a new identical model.
            if self.store_intermediate_classifiers:
                self._classifiers.append(self._classifier)
                self._classifier = sk.base.clone(self._classifier)
        return self.errors
