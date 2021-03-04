import datetime
import pandas as pd
import random
import string
import time
import typing


def get_random_string(length: int) -> str:
    """
    This function returns a random string of letters of a specified length.

    Args:
        length (int): length of the random string to generate

    Returns:
        rand_str (str): Random string.
    """
    candidates = string.ascii_letters + string.digits
    rand_str = ''.join(random.choice(candidates)
                       for i in range(length))
    return rand_str


def get_timestamp_as_string() -> str:
    """
    This function returns a timestamp as a writable string.

    Returns:
        ts (str): String with the current time and date format as %Y%m%d-%H%M%S.
    """
    ts = time.strftime('%Y%m%d-%H%M%S')
    return ts


def remove_zero_fare_and_oob_rows(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    This function removes rows with zero-valued fare amounts and out of bounds of the start and end dates.

    Args:
        df: pd dataframe representing data
        start_date (optional): minimum date in the resulting dataframe
        end_date (optional): maximum date in the resulting dataframe

    Returns:
        pd: DataFrame representing the cleaned dataframe
    """
    df = df[df.fare_amount > 0]  # avoid divide-by-zero
    if start_date:
        df = df[df.tpep_dropoff_datetime.astype('str') >= start_date]
    if end_date:
        df = df[df.tpep_dropoff_datetime.astype('str') <= end_date]

    return df.reset_index(drop=True)


def get_file_at_latest_timestamp(filenames: typing.List[str]) -> str:
    """
    This file takes in a list of filenames and returns the filename with the latest timestamp.

    Args:
        filenames (List[str]): list of string filenames

    Returns:
        str corresponding to the filename with the latest timestamp
    """
    valid_filenames = []

    # Disregard filenames that aren't of the timestamp format
    format = '%Y%m%d-%H%M%S'
    for filename in filenames:
        try:
            # Strip down to the suffix without the .pq extension
            datetime.datetime.strptime(filename.split('/')[-1][:-3], format)
            valid_filenames.append(filename)
        except ValueError:
            pass

    return max(valid_filenames)
