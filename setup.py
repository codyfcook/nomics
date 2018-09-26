#!/usr/bin/env python
from __future__ import absolute_import
from setuptools import setup, find_packages

__version__ = '0.1.0'

  
def requirements(*filenames):
    """Takes all input requirements files and pulls out modules."""

    requires = []

    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    # skip blank lines and comments
                    continue
                requires.append(line)

    return requires

def readme():
    with open('README.md') as f:
        return f.read()



setup(
        name='nomics',
        version=__version__,
        description='Custom functions for ubernomics work',
        url='https://code.uberinternal.com/diffusion/UBNOM',
        author='Cody Cook',
        author_email='cook@uber.com',
        license='MIT',
        packages=find_packages(),
        install_requires=requirements('requirements.txt'),
        include_package_data=True,
        long_description=readme()
    )
