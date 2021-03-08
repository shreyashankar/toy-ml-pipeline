import calendar
import itertools
import os
import pandas as pd

from utils import io, helpers

if __name__ == '__main__':
    months = [f'{month:02d}' for month in range(1, 13)]
    years = ['2020']
    product = itertools.product(months, years)

    for month, year in product:
        raw_df = io.read_file(month, year)

        first, last = calendar.monthrange(int(year), int(month))
        first_day = f'{year}-{month}-{first:02d}'
        last_day = f'{year}-{month}-{last:02d}'
        clean_df = helpers.remove_zero_fare_and_oob_rows(
            raw_df, first_day, last_day)

        # Write "clean" df to s3
        component = os.path.join('clean', f'{year}_{month}')
        print(io.save_output_df(clean_df, component))
