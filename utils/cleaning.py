import pandas as pd


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
