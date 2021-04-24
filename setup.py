#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='q10hybrid',
    version='0.0.0',
    description='Hybrid modeling of ecosystem respiration temperature sensiticity',
    author='Basil Kraft',
    author_email='basilkraft@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/bask0/q10hybrid',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
