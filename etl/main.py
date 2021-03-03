import pandas as pd

from utils import io, cleaning
import os

if __name__ == '__main__':
    # Load January 2020
    raw_df = io.read_file('01', '2020')
    clean_df = cleaning.remove_zero_fare_and_oob_rows(
        raw_df, '2020-01-01', '2020-01-31')

    # Write "clean" df to s3
    print(io.save_output(clean_df, 'clean'))
