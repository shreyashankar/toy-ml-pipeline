import pandas as pd

from utils import io

if __name__ == '__main__':
    # Load latest features
    df = io.load_output_df('features')

    # Parameters
    num_rows = len(df.index)
    train_frac = 0.8
    component_prefix = 'training/files'

    # Split into train and test based on time
    df = df.sort_values(by=['tpep_pickup_datetime'], ascending=True)
    train_df = df.head(int(train_frac * num_rows))
    test_df = df.tail(int((1 - train_frac) * num_rows))

    # Save train and test sets
    print(io.save_output_df(train_df, f'{component_prefix}/train'))
    print(io.save_output_df(test_df, f'{component_prefix}/test'))
