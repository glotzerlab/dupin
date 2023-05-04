.. highlight:: shell

============
Installation
============


From source
------------

The sources for event-detection can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/glotzerlab/dupin

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/glotzerlab/dupin/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/glotzerlab/dupin
.. _tarball: https://github.com/glotzerlab/dupin/tarball/master


Building Documentation
----------------------

Currently the documentation is not available online, but can be built locally. The required packages
are

+ sphinx

These can be installed with ``python3 -m pip install sphinx``. To build documentation in the project
base directory run ``python3 -m sphinx ./docs ./docs/_build``. To view the built documentation open
the ``index.html`` file in ``./docs/_build`` with your preferred browser.
