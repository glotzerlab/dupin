#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("changelog") as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "ruptures",
    "kneed",
    "freud-analysis",
    "numba",
    "scikit-learn",
]

setup(
    author="Brandon Butler",
    author_email="butlerbr@umich.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Python package for detecting rare events in molecular "
    "simulations.",
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="event_detection",
    name="event_detection",
    packages=find_packages(),
    url="https://github.com/b-butler/event_detection",
    version="0.0.1",
    zip_safe=False,
)
