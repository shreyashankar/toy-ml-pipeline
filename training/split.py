from mltrace import get_db_uri, set_db_uri, create_component, tag_component, register
from utils import io

import os
import pandas as pd


@register(component_name='split', input_vars=['input_files'], output_vars=['train_output_path', 'test_output_path'])
def main():
    train_months = ['2020_01']
    test_month = '2020_02'

    input_files = [io.get_output_path(os.path.join(
        'features', month)) for month in train_months + [test_month]]

    # Load latest features for train set
    train_df = pd.concat(
        [io.load_output_df(os.path.join('features', month)) for month in train_months])

    # Load latest features for test set
    test_df = io.load_output_df(os.path.join('features', test_month))

    # Save train and test sets
    component_prefix = 'training/files'
    train_output_path = io.save_output_df(train_df, f'{component_prefix}/train')
    test_output_path = io.save_output_df(test_df, f'{component_prefix}/test')

    print(train_output_path)
    print(test_output_path)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', '54.177.215.161'))

    # Create components
    create_component(
        'split', 'Splitting features into train and test sets.', 'shreya')
    tag_component('split', ['training'])

    main()
