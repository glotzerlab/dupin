===============
event-detection
===============

Python package for detecting rare events in molecular simulations.


Conceptual Design
-----------------

``event-detection`` is based around a five step model going from a raw simulation in the form of a trajectory into a set of change points defining the boundaries of events in a molecular simulation. The steps are

1. Data Collection - A variety of features need to be calculated in order to feed into the change point detection algorithms. These should cover any potential changes to a system that can be anticipated. Each feature can be a global value or a vector describing the system.
2. Filters and Reducers - Many desirable features occur on a per particle level such as Minkowski structure metrics, but for change point detection, having a feature for each particle is not desirable, so reducers provide an avenue to take a per particle quantity and transform it into a set number scalar features. These reducers rely on the fact that only a few features of the distribution are necessary to detect a transition (namely those at the extremes of a distribution).
3. Signal Filtering - Given thermodynamic noise and other processes, the signal of a feature across a trajectory may carry noise that is not desired for detection purposes. This is highly cost function dependent. For example, some cost functions such as those that rely on the shift of the standard deviation of a variable are sensitive to the scale of noise and will have reduced efficacy applied to a signal with dampened noise. However, other cost functions, may perform worse with a high noise signal (that is thermodynamically relevant) and could benefit from a form of signal filtering.
4. Event Detection - With properly reduced and/or filtered features, event detection algorithms can be used to discern where events occur within the simulation. This may be the final stage of the pipeline, but there is a potential fifth step.
5. Elbow Detection - When the number of change points are not known and the appropriate penalty to use for a given change point detection algorithm is not known, using fixed cardinality solvers across a range of number of change points combined with elbow detection, allows for the automatic determination of the number of change points for a given simulation.

All steps besides 1 and 4 are optional depending on the circumstances. For instance, if using the Nematic order parameter, there would be no need for step 2.


Features
--------

* TODO

Credits
-------

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the
``b-butler/my-pypackage`` project template which is based on
`audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`_.
