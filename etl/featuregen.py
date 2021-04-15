from mltrace import get_db_uri, set_db_uri, create_component, tag_component, log_component_run, get_git_hash
from mltrace.entities import ComponentRun
from utils import io, feature_generators

import itertools
import os
import pandas as pd


def featurize_data(month: str, year: str) -> str:
    # Load latest clean data
    clean_component = os.path.join('clean', f'{year}_{month}')
    df = io.load_output_df(clean_component)
    input_path = io.get_output_path(clean_component)

    # Logging
    cr = ComponentRun('featuregen')
    cr.add_input(input_path)
    cr.set_start_timestamp()
    cr.git_hash = get_git_hash()

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
    output_path = io.save_output_df(features_df, features_component)

    cr.set_end_timestamp()
    cr.add_output(output_path)
    log_component_run(cr)

    return output_path


def main():
    months = [f'{month:02d}' for month in range(1, 13)]
    years = ['2020']
    product = itertools.product(months, years)

    for month, year in product:
        print(featurize_data(month, year))


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', 'localhost'))

    # Create components
    create_component(
        'featuregen', 'Generates features from the clean data.', 'shreya')
    tag_component('featuregen', ['etl'])

    main()
