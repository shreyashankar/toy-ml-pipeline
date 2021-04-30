from mltrace import get_db_uri, set_db_uri, create_component, tag_component, register
from utils import io, models

import pandas as pd


@register('train', input_vars=['train_file_path', 'test_file_path'], output_vars=['output_path'])
def main():
    # Load train set
    base = 'training/files'
    train_df = io.load_output_df(f'{base}/train')
    train_file_path = io.get_output_path(f'{base}/train')
    test_df = io.load_output_df(f'{base}/test')
    test_file_path = io.get_output_path(f'{base}/test')

    feature_columns = [
        'pickup_weekday', 'pickup_hour', 'pickup_minute', 'work_hours',
        'passenger_count', 'trip_distance', 'trip_time', 'trip_speed',
        'PULocationID', 'DOLocationID', 'RatecodeID'
    ]
    label_column = 'high_tip_indicator'

    model_params = {
        'max_depth': 10
    }

    # Create and train model
    mw = models.RandomForestModelWrapper(
        feature_columns=feature_columns, model_params=model_params)
    mw.train(train_df, label_column)

    # Score model
    train_score = mw.score(train_df, label_column)
    test_score = mw.score(test_df, label_column)

    mw.add_data_path('train_df', train_file_path)
    mw.add_data_path('test_df', test_file_path)
    mw.add_metric('train_f1', train_score)
    mw.add_metric('test_f1', test_score)

    # Print paths and metrics
    print('Paths:')
    print(mw.get_data_paths())
    print('Metrics:')
    print(mw.get_metrics())

    # Print feature importances
    print(mw.get_feature_importances())

    # Save model
    output_path = mw.save('training/models')
    print(output_path)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', '54.177.215.161'))

    # Create components
    create_component(
        'train', 'Training a model.', 'shreya')
    tag_component('train', ['training'])

    main()
