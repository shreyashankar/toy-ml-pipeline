"""
inference_batch.py

Sample script that runs inference on a batch. It pings the "api" (served locally in inference/app.py) and gets a response. To use, make sure you are also running inference/app.py.
"""

from utils import io

import pandas as pd
import requests


def main():
    # Grab latest features and model wrapper. Sort features by date.
    df = io.load_output_df('features/2020_12')\
           .sort_values(by=['tpep_pickup_datetime'], ascending=True)\
           .drop('tpep_pickup_datetime', axis=1)

    # Run model on latest features for a random example
    url = 'http://127.0.0.1:8000/predict'
    
    req = {'data': df.to_json(), 'feature_path': io.get_output_path('features/2020_12')}

    response = requests.post(
        url, json=req)
    # print(f'Response: {response.json()}')

    print('Exiting.')


if __name__ == '__main__':
    main()