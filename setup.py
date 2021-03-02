from setuptools import setup, find_packages

setup(
    name='toy-applied-ml-pipeline',
    version='0.1',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'fsspec',
        'numpy',
        'pandas',
        's3fs'
    ]
)
