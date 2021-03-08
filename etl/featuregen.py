import itertools
import os
import pandas as pd

from utils import io, feature_generators

if __name__ == '__main__':
    months = [f'{month:02d}' for month in range(1, 13)]
    years = ['2020']
    product = itertools.product(months, years)

    for month, year in product:
        # Load latest clean data
        clean_component = os.path.join('clean', f'{year}_{month}')
        df = io.load_output_df(clean_component)

        # Create features and label
        pickup_features = feature_generators.Pickup().compute(df)
        trip_features = feature_generators.Trip().compute(df)
        categorical_features = feature_generators.Categorical().compute(df)
        label = feature_generators.HighTip().compute(df)

        # Concatenate features
        features_df = pd.concat(
            [pickup_features, trip_features, categorical_features, label, df['tpep_pickup_datetime'].to_frame()], axis=1)

        # Write features to s3
        features_component = os.path.join('features', f'{year}_{month}')
        print(io.save_output_df(features_df, features_component))
