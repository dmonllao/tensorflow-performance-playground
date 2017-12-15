# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Version read from file.
version = '0.0.1'

setup(
    name='tensorflow-performance-playground',

    version=version,

    description='Tensorflow playground for comparing different algorithms performance.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/dmonllao/tensorflow-performance-playground',

    # Author details
    author='David Monllao',
    author_email='david.monllao@gmail.com',

    # Choose your license
    license='GPLv3',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='moodle machine learning numpy scikit-learn tensorflow',

    packages=find_packages(),
    package_data={
        'moodlemlbackend': ['VERSION']
    },
    install_requires=[
        'numpy>=1.11.0,<1.12',
        'pandas>=0.17.0',
        'scikit-learn>=0.17.0,<0.18',
        'scipy>=0.17.0,<0.18',
        'tensorflow>=1.4',
    ],
)
