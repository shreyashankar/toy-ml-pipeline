"""
io.py

This file contains helper functions for reading and writing files.
"""
import pandas as pd


def read_file(month: str, year: str) -> pd.DataFrame:
    """
    This function reads from the NYC taxicab public s3 bucket to get a dataframe of all yellow trips for the specified month and year.

    Args:
        month (str): month formatted as a 2-digit string, i.e. "05" for May
        year (str): year formatted as a 4-digit string, i.e. "2020" for 2020

    Returns:
        pd.DataFrame: dataframe corresponding to all the yellow taxicab trips for the given parameters
    """

    assert len(month) == 2, "Month must be a 2-digit string."
    assert len(year) == 4, "Year must be a 4-digit string."

    base_path = 's3://nyc-tlc/trip data/yellow_tripdata'
    file_path = f'{base_path}_{year}-{month}.csv'

    # Load data
    df = pd.read_csv(
        file_path,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"], memory_map=True)

    return df
