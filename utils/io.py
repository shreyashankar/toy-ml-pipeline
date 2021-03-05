"""
io.py

This file contains helper functions for reading and writing files.
"""
import boto3
import os
import pandas as pd
import pickle
import s3fs
import typing

from .helpers import *

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


def create_output_path(component: str, dev: bool = True, version: str = None) -> str:
    """
    This function creates a filename for the output of a pipeline component (a dataframe).

    Args:
        component (str): name of the component that produced the output (ex: clean)
        dev (bool, optional): whether this is run in development or "production" mode
        version (str, optional): optional version for the output. If not specified, the function will create the version number.

    Returns:
        output_path (str): Full path for the file
    """
    assert len(component) > 0, 'Component name should not be empty.'

    prefix = os.path.join(
        'dev', component) if dev else os.path.join('prod', component)

    assert list_files(
        prefix), 'Component does not exist. Specify the correct component or contact an administrator to create it.'

    # Create version if doesn't exist
    if version == None:
        version = get_timestamp_as_string()

    output_path = os.path.join(prefix, version)

    return output_path


def save_output_pkl(obj: object, component: str, dev: bool = True, overwrite: bool = False, version: str = None) -> str:
    """
    This function serializes an object as part of a component's output.

    Args:
        obj (object): Python object
        component (str): name of the component that produced the output (ex: clean)
        dev (bool, optional): whether this is run in development or "production" mode
        overwrite (bool, optional): whether to overwrite a file with the same name
        version (str, optional): optional version for the output. If not specified, the function will create the version number.

    Returns:
        path (str): Full path that the file can be accessed at
    """

    output_path = create_output_path(component, dev, version)
    filename = f'{output_path}.pkl'

    # Make sure file doesn't exist if overwrite is False
    if list_files(filename) and overwrite is False:
        raise OSError(
            'Trying to overwrite a with this component name and version. Please try another name / version or set overwrite to True.')

    pkl_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET_NAME, filename).put(Body=pkl_obj)
    filename = os.path.join(f's3://{BUCKET_NAME}', filename)
    return filename


def save_output_df(df: pd.DataFrame, component: str, dev: bool = True, overwrite: bool = False, version: str = None) -> str:
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

    output_path = create_output_path(component, dev, version)
    filename = f'{output_path}.pq'

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


def load_output_df(component: str, dev: bool = True, version: str = None) -> pd.DataFrame:
    """
    This function loads the latest version of data that was produced by a component.

    Args:
        component (str): component name that we want to get the output from
        dev (bool): whether this is run in development or "production" mode
        version (str, optional): specified version of the data

    Returns:
        df (pd.DataFrame): dataframe corresponding to the data in the latest version of the output for the specified component
    """

    # Load data
    filename = get_output_path(component, dev, version)
    df = pd.read_parquet(f's3://{filename}')
    return df


def load_output_pkl(component: str, dev: bool = True, version: str = None) -> object:
    """
    This function loads the latest version of an object that was produced by a component.

    Args:
        component (str): component name that we want to get the output from
        dev (bool): whether this is run in development or "production" mode
        version (str, optional): specified version of the data

    Returns:
        object: object corresponding to the latest version of the output for the specified component
    """
    filename = get_output_path(component, dev, version)

    # Strip bucket name
    filename = '/'.join(filename.split('/')[1:])
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
    serialized_obj = obj['Body'].read()
    deserialized_obj = pickle.loads(serialized_obj)
    return deserialized_obj


def get_output_path(component: str, dev: bool = True, version: str = None) -> str:
    """
    This function gets the path corresponding to the latest or specified version of a component.

    Args:
        component (str): component name that we want to get the output from
        dev (bool): whether this is run in development or "production" mode
        version (str, optional): specified version of the data

    Returns:
        filename: filename corresponding the specified or latest version of the output for the specified component
    """

    assert len(component) > 0, 'Component name should not be empty.'

    prefix = os.path.join(
        'dev', component) if dev else os.path.join('prod', component)

    if version is not None:
        prefix = os.path.join(prefix, version)

    filenames = list_files(prefix)
    filename = get_file_at_latest_timestamp(filenames)

    return filename
