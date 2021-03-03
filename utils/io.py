"""
io.py

This file contains helper functions for reading and writing files.
"""
import os
import pandas as pd
import s3fs
import typing

from .naming import *

BUCKET_NAME = 'toy-applied-ml-pipeline'


def read_file(month: str, year: str) -> pd.DataFrame:
    """
    This function reads from the NYC taxicab public s3 bucket to get a dataframe of all yellow trips for the specified month and year.

    Args:
        month (str): month formatted as a 2-digit string, i.e. "05" for May
        year (str): year formatted as a 4-digit string, i.e. "2020" for 2020

    Returns:
        pd.DataFrame: dataframe corresponding to all the yellow taxicab trips for the given parameters
    """

    assert len(month) == 2, 'Month must be a 2-digit string.'
    assert len(year) == 4, 'Year must be a 4-digit string.'

    base_path = 's3://nyc-tlc/trip data/yellow_tripdata'
    file_path = f'{base_path}_{year}-{month}.csv'

    # Load data
    df = pd.read_csv(
        file_path,
        parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], memory_map=True)

    return df


def write_file(df: pd.DataFrame, suffix: str, scratch: bool = True) -> str:
    """
    This function takes a dataframe and writes it to the path specified in the args, without the index, as a parquet file.

    Args:
        df (pd.DataFrame): Pandas DataFrame to write to S3 as a parquet file
        suffix (str): path to add to the S3 bucket prefix. Must end with ".pq" or ".parquet"
        scratch (bool): whether the path should be prefixed with scratch

    Returns:
        path (str): Full path that the file can be accessed at
    """

    assert suffix.endswith('.parquet') or suffix.endswith(
        '.pq'), 'Path suffix supplied must end with .pq or .parquet'

    path = f's3://{BUCKET_NAME}'
    if scratch:
        path = os.path.join(path, 'scratch')
    path = os.path.join(path, suffix)

    df.to_parquet(path, index=False)
    return path


def save_output(df: pd.DataFrame, component: str, dev: bool = True, overwrite: bool = False, version: str = None) -> str:
    """
    This function writes the output of a pipeline component (a dataframe) to a parquet file.

    Args:
        df (pd.DataFrame): dataframe representing the output
        component (str): name of the component that produced the output (ex: clean)
        dev (bool, optional): whether this is run in development or "production" mode
        overwrite (bool, optional): whether to overwrite a file with the same name
        version (str, optional): optional version for the output. If not specified, the function will create the version number.

    Returns:
        path (str): Full path that the file can be accessed at
    """

    assert len(component) > 0, 'Component name should not be empty.'

    prefix = os.path.join(
        'dev', component) if dev else os.path.join('prod', component)

    assert list_files(
        prefix), 'Component does not exist. Specify the correct component or contact an administrator to create it.'

    # Create version if doesn't exist
    if version == None:
        version = get_timestamp_as_string()

    filename = os.path.join(prefix, f'{version}.pq')

    # Make sure file doesn't exist if overwrite is False
    if list_files(filename) and overwrite is False:
        raise OSError(
            'Trying to overwrite a with this component name and version. Please try another name / version or set overwrite to True.')

    return write_file(df, filename, scratch=False)


def list_files(prefix: str = "") -> typing.List[str]:
    """
    Given a prefix, this function returns a list of all the immediate files in that directory, (1 level deep).

    Args:
        prefix (str): folder or directory to search, relative to the bucket name.

    Returns:
        List[str]: list of strings corresponding to the immediate files in the directory specified.
    """
    fs = s3fs.S3FileSystem(anon=True)
    path = os.path.join(BUCKET_NAME, prefix)
    return fs.ls(path)
