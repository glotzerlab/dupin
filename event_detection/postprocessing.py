"""General functions for analyzing change points once detected."""


import event_detection.preprocessing


def compute_features_in_event(signal, change_points, sample_size, sensitivity):
    """Compute the participating features within a pair of change points.

    Warning:
        This function is designed to work with a linear cost function, and
        assumes change points denote differences in linear signals (e.g. not
        mean-shift signals).

    Internally this function uses
    `event_detection.preprocessing.filter.mean_shift`.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` numpy.ndarray of float
        The signal the change points are from.
    change_points: list[int]
        A list of all interior change points (beginning and end of signal not
        included).
    sample_size: float or int, optional
        Either the fraction of the overall signal to use to evaluate the
        statistics of each end of all subsignals defined by the change points,
        or the number of data points to use on each end of the signal for
        statistics. Default to 0.1. If this would result in less than three data
        points, three will be used.
    sensitivity: float, optional
        The minimum likelihood that one of the signal's end's mean is drawn from
        the Gaussian approximation of the other end to require. In other words,
        the lower the number the increased probability that the difference in
        means is not random. Defaults to 0.01.

    Returns
    -------
    participating_features: list [`numpy.ndarray` of float]
        Returns a list of Boolean arrays that filter the original data into
        participating features during each interval. A value of ``None`` is used
        for all intervals that are too small to analyze.
    """
    augmented_change_points = [0] + change_points + [len(signal)]
    participating_features = []
    for beg, end in zip(augmented_change_points, augmented_change_points[1:]):
        try:
            section_features = event_detection.preprocessing.filter.mean_shift(
                signal[beg:end], sample_size, sensitivity, return_filter=True
            )
        # If signal is too small, then we just append None as an indicator.
        except ValueError:
            participating_features.append(None)
        else:
            participating_features.append(section_features)
    return participating_features
