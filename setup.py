#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirement_list = requirements_file.readlines()

requirements = requirement_list
#requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Archit Nikhil Datar",
    author_email='architdatar@gmail.com',
    python_requires='>=3.9 <=3.12',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Uncertainty quantification and model inference for machine learning models",
    entry_points={
        'console_scripts': [
            'ml_uncertainty=ml_uncertainty.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ml_uncertainty',
    name='ml_uncertainty',
    packages=find_packages(include=['ml_uncertainty', 'ml_uncertainty.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/architdatar/ml_uncertainty',
    version='0.1.0',
    zip_safe=False,
)
