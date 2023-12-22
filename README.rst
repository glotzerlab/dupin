=====
dupin
=====

|Cite|
|Github-Stars|

.. |Cite| image:: https://img.shields.io/badge/dupin-cite-yellow
   :target: https://dupin.readthedocs.io/citing.html
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/glotzerlab/dupin.svg
   :target: https://github.com/glotzerlab/dupin

Welcome to ``dupin`` a Python package for detecting rare events in molecular simulations.

Overview
--------

**dupin** is designed to provide an unopinionated Python API for partitioning temporal point cloud data into regions of stability and transition.
Generally such data comes from molecular simulations or experimental imaging techniques.
For example, if a researcher imaged gold nanoparticles nucleating into FCC crystal, **dupin** could help to partition the system into the initial fluid, transition, and crystal regions of the point cloud trajectory.
Though, **dupin** attempts to be general and *unopinionated* we provide sensible defaults and do not sacrifice ease of use for customizability.

Resources
---------
- `Documentation <https://dupin.readthedocs.io/en/latest/index.html>`__: Read our API documentation.
- `Installation Guide <https://dupin.readthedocs.io/en/latest/installation.html>`__: Look at our installation guide.
- `GitHub Repository <https://github.com/glotzerlab/dupin>`__: Check out the source code.
- `Issue Tracker <https://github.com/glotzerlab/dupin/issues>`__: File a bug report or pull request.

Related Tools
-------------

- `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable/index.html>`__: Molecular simulation engine
- `freud <https://freud.readthedocs.io/en/stable/index.html>`__: Molecular trajectory analysis
- `ruptures <https://centre-borelli.github.io/ruptures-docs/>`__: Change point decection library
- `kneed <https://kneed.readthedocs.io/en/latest/>`__: Elbow detection library

Example
-------

.. code-block:: python

    import dupin as du
    import numpy as np
    import ruptures as rpt

    signal = np.load("signal.npy")
    dynp = rpt.Dynp(custom_cost=du.detect.CostLinearFit())
    detector = du.detect.SweepDetector(dynp, 8)

    chps = detector.fit(signal)

.. image:: docs/_static/detect.gif

Credits
-------

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ based on a
modified version of `audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`_.
