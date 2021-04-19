from mltrace import get_db_uri, set_db_uri, create_component, register, tag_component
from utils import io, helpers

import calendar
import itertools
import os
import pandas as pd


@register('cleaning', input_vars=['raw_data_filename'], output_vars=['output_path'])
def clean_data(raw_data_filename: str, raw_df: pd.DataFrame, month: str, year: str, component: str) -> str:
    first, last = calendar.monthrange(int(year), int(month))
    first_day = f'{year}-{month}-{first:02d}'
    last_day = f'{year}-{month}-{last:02d}'
    clean_df = helpers.remove_zero_fare_and_oob_rows(
        raw_df, first_day, last_day)

    # Write "clean" df to s3
    output_path = io.save_output_df(clean_df, component)

    return output_path


def main():
    months = [f'{month:02d}' for month in range(1, 13)]
    years = ['2020']
    product = itertools.product(months, years)

    for month, year in product:
        print(f'Reading {month}-{year}...')
        raw_df = io.read_file(month, year)
        raw_data_filename = io.get_raw_data_filename(month, year)
        component = os.path.join('clean', f'{year}_{month}')
        output_path = clean_data(
            raw_data_filename, raw_df, month, year, component)
        print(output_path)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', 'localhost'))

    # Create components
    create_component(
        'cleaning', 'Cleans the raw data with basic OOB criteria.', 'shreya')
    tag_component('cleaning', ['etl'])

    main()
