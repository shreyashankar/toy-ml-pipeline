from mltrace import get_db_uri, set_db_uri, create_component, tag_component, log_component_run, get_git_hash
from mltrace.entities import ComponentRun
from utils import io

import os
import pandas as pd


def main():
    train_months = ['2020_09', '2020_10']
    test_month = '2020_11'

    input_files = [io.get_output_path(os.path.join(
        'features', month)) for month in train_months + [test_month]]

    # Logging
    cr = ComponentRun('split')
    cr.add_inputs(input_files)
    cr.set_start_timestamp()
    cr.git_hash = get_git_hash()

    # Load latest features for train set
    train_df = pd.concat(
        [io.load_output_df(os.path.join('features', month)) for month in train_months])

    # Load latest features for test set
    test_df = io.load_output_df(os.path.join('features', test_month))

    # Save train and test sets
    component_prefix = 'training/files'
    train_output_path = io.save_output_df(train_df, f'{component_prefix}/train')
    test_output_path = io.save_output_df(test_df, f'{component_prefix}/test')

    cr.add_outputs([train_output_path, test_output_path])
    cr.set_end_timestamp()
    log_component_run(cr)

    print(train_output_path)
    print(test_output_path)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', 'localhost'))

    # Create components
    create_component(
        'split', 'Splitting features into train and test sets.', 'shreya')
    tag_component('split', ['training'])

    main()
