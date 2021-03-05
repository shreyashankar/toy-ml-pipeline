import pandas as pd

from utils import io, models

if __name__ == '__main__':
    # Load train set
    df = io.load_output_df('training/files/train')
    train_file_path = io.get_output_path('training/files/train')

    feature_columns = [
        'pickup_weekday', 'pickup_hour', 'pickup_minute', 'work_hours',
        'passenger_count', 'trip_distance', 'trip_time', 'trip_speed',
        'PULocationID', 'DOLocationID', 'RatecodeID'
    ]
    label_column = 'high_tip_indicator'

    model_params = {
        'max_depth': 10
    }

    mw = models.RandomForestModelWrapper(
        feature_columns=feature_columns, model_params=model_params)
    mw.train(df, label_column)
    score = mw.score(df, label_column)
    mw.add_data_path('train_df', train_file_path)
    mw.add_metric('f1', score)

    # Print paths and metrics
    print('Paths:')
    print(mw.get_data_paths())
    print('Metrics:')
    print(mw.get_metrics())

    # Save model
    print(mw.save('training/models'))
