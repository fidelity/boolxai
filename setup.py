# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import os

import setuptools

# Note: execute _version.py since we cannot do this:
# from boolxai._version import __author__, __docurl__, __url__, __version__
# because it imports boolxai which will fail due to not having installed the
# dependencies yet!
with open(os.path.join("boolxai", "_version.py")) as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements_plot.txt") as fh:
    plot_reqs = fh.read().splitlines()

setuptools.setup(
    name="boolxai",
    description="BoolXAI: Explainable AI using expressive Boolean formulas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author=__author__,
    url=__url__,
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=required,
    python_requires=">=3.8",
    extras_require={"plot": plot_reqs},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": __url__,
        "Documentation": __docurl__,
    },
)
