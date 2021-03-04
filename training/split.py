import pandas as pd

from utils import io

if __name__ == '__main__':
    # Load latest features
    df = io.load_output('features')

    # Parameters
    num_rows = len(df.index)
    train_frac = 0.8
    component = 'training/files'

    # Split into train and test based on time
    df = df.sort_values(by=['tpep_pickup_datetime'], ascending=True)
    train_df = df.head(int(train_frac * num_rows))
    test_df = df.tail(int((1 - train_frac) * num_rows))

    # Save train and test sets
    print(io.save_output(train_df, f'{component}/train'))
    print(io.save_output(test_df, f'{component}/test'))
