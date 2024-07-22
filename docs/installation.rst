.. highlight:: shell

============
Installation
============

**dupin** is available on `PyPI`_ and `conda-forge`_. Alternatively, users can also
install **dupin** from source.

Install via conda
-----------------

**dupin** is available on conda-forge_. Install with:

.. code:: bash

   mamba install dupin

Install via pip
-----------------

**dupin** is also available from PyPI_. To install **dupin** into a *non-conda* virtual
environment (if uv_ is installed), execute:

.. code:: bash

   uv pip install dupin

or

.. code:: bash

   python3 -m pip install dupin

.. _conda-forge: https://conda-forge.org/
.. _PyPI: https://pypi.org/
.. _uv: https://github.com/astral-sh/uv


From source
------------

The sources for event-detection can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/glotzerlab/dupin

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/glotzerlab/dupin/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install /path/to/clone


.. _Github repo: https://github.com/glotzerlab/dupin
.. _tarball: https://github.com/glotzerlab/dupin/tarball/main


Building Documentation
----------------------

Currently the documentation is not available online, but can be built locally.
The required packages are

+ furo
+ sphinx
+ nbsphinx

These can be installed with ``python3 -m pip install sphinx nbsphinx furo``.
To build documentation in the project base directory run ``python3 -m sphinx ./docs ./docs/_build``.
To view the built documentation open the ``index.html`` file in ``./docs/_build`` with your preferred browser.
