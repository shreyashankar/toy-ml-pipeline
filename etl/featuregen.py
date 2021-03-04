import pandas as pd

from utils import io, feature_generators

if __name__ == '__main__':
    # Load latest clean data
    df = io.load_output('clean')

    # Create features and label
    pickup_features = feature_generators.Pickup().compute(df)
    trip_features = feature_generators.Trip().compute(df)
    categorical_features = feature_generators.Categorical().compute(df)
    label = feature_generators.HighTip().compute(df)

    # Concatenate features
    features_df = pd.concat(
        [pickup_features, trip_features, categorical_features, label, df['tpep_pickup_datetime'].to_frame()], axis=1)

    # Write features to s3
    print(io.save_output(features_df, 'features'))
