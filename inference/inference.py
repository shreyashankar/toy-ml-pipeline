"""
inference.py

Sample script that runs inference on examples incrementally. It pings the "api" (served locally in inference/app.py) and gets a response. To use, make sure you are also running inference/app.py.
"""

from utils import io

import pandas as pd
import requests


def main():
    # Grab latest features and model wrapper. Sort features by date.
    df = io.load_output_df('features/2020_03')\
           .sort_values(by=['tpep_pickup_datetime'], ascending=True)\
           .drop('tpep_pickup_datetime', axis=1)
    feature_path = io.get_output_path('features/2020_03')

    # Run model on latest features for a random example
    url = 'http://127.0.0.1:8000/predict'
    idx = 0
    preds = []

    while(True):
        inp = input('Press enter to make a prediction and q to quit.\n')
        if (inp.strip().lower() == 'q'):
            break
        example = df.iloc[[idx]]
        print(f'Request: {example}')

        req = {
            'data': example.to_json(),
            'row_idx': idx,
            'feature_path': feature_path
        }

        response = requests.post(
            url, json=req)
        print(f'Response: {response.json()}')
        idx += 1
        preds.append(response.json()['prediction'][0])

    print('Exiting.')


if __name__ == '__main__':
    main()
