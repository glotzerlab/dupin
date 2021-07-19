============
Contributing
============

Contributions are welcomed via [pull requests on GitHub](https://github.com/glotzerlab/b-butler/event-detection).
Create an issue first to ensure that the proposed changes are in line with the direction of the
package.

Features
========

Implement functionality in a general and flexible fashion
---------------------------------------------------------

New features should be applicable to a variety of use-cases.

Propose a minimal set of related changes
----------------------------------------

All changes in a pull request should be closely related. Multiple change sets that
are loosely coupled should be proposed in separate pull requests.

Agree to the Contributor Agreement
----------------------------------

All contributors must agree to the Contributor Agreement ([ContributorAgreement.md](ContributorAgreement.md)) before their pull request can be merged.

Source code
===========

This package uses pre-commit to ensure consistent formatting of Python code. To install
``pre-commit`` run the following commands in the project directory.

.. code:: shell

    python3 -m pip install pre-commit
    pre-commit install

Document code with comments
---------------------------

Write proper docstrings for all modules, classes, and functions.
In addition write standard ``# comments`` when the code is complicated or such comments would
improve understanding.

Tests
=====

Write unit tests
----------------

Add unit tests for all new functionality.

Add developer to the AUTHORS.rst
--------------------------------

Update the credits documentation to reference what each developer contributed to the code.

Propose a change log entry
--------------------------

Propose a short concise entry describing the change in the pull request description.
