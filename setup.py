from setuptools import setup, find_packages

setup(
    name='toy-applied-ml-pipeline',
    version='0.1',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'boto3',
        'flask',
        'fsspec',
        'numpy',
        'pandas',
        'pyarrow',
        'pytest',
        'requests',
        's3fs',
        'sklearn'
    ],
    entry_points={
        'console_scripts': [
            'cleaning=etl.cleaning:main',
            'featuregen=etl.featuregen:main',
            'split=training.split:main',
            'train=training.train:main',
            'serve=inference.app:main',
            'inference=inference.inference:main'
        ],
    }
)
