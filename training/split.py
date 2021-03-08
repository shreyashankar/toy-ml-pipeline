from utils import io

import os
import pandas as pd

if __name__ == '__main__':
    train_months = ['2020_10', '2020_11']
    test_month = '2020_12'

    # Load latest features for train set
    train_df = pd.concat(
        [io.load_output_df(os.path.join('features', month)) for month in train_months])

    # Load latest features for test set
    test_df = io.load_output_df(os.path.join('features', test_month))

    # Save train and test sets
    component_prefix = 'training/files'
    print(io.save_output_df(train_df, f'{component_prefix}/train'))
    print(io.save_output_df(test_df, f'{component_prefix}/test'))
