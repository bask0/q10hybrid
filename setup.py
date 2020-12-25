#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='Describe Your Project',
    author='Basil Kraft',
    author_email='basilkraft@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/bask0/dl_template',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
