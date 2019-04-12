#!/usr/bin/env python
from setuptools import setup, find_packages

from pandas_implants import __version__

options = dict(
    name='pandas_implants',
    version=__version__,
    packages=find_packages(),
    license='MIT',
    description='Experimental types for pandas.',
    author='Jan Pipek',
    author_email='jan.pipek@gmail.com',
    url='https://github.com/janpipek/pandas_implants',
    install_requires = ['pandas'],
    python_requires=">=3.6",
)

setup(**options)