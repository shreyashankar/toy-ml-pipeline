"""
inference.py

Sample script that runs inference on examples incrementally. It pings the "api" (served locally in inference/app.py) and gets a response. To use, make sure you are also running inference/app.py.
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
    url = 'http://localhost:5000/predict'
    idx = 0
    preds = []

    while(True):
        inp = input('Press enter to make a prediction and q to quit.\n')
        if (inp.strip().lower() == 'q'):
            break
        example = df.iloc[[idx]].to_dict('r')[0]
        idx += 1
        print(f'Request: {example}')

        response = requests.post(
            url, json=example)
        print(f'Response: {response.json()}')
        preds.append(responds['prediction'])

    print('Exiting.')


if __name__ == '__main__':
    main()
