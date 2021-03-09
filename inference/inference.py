"""
inference.py

Sample script that runs inference on random examples in December 2020. It pings the "api" (served locally in inference/app.py) and gets a response.
"""

from utils import io

import pandas as pd
import requests

if __name__ == '__main__':
    # Grab latest features and model wrapper
    df = io.load_output_df(
        'features/2020_12').drop('tpep_pickup_datetime', axis=1)

    # Run model on latest features for a random example
    url = 'http://localhost:5000/predict'

    while(True):
        inp = input('Press enter to make a prediction and q to quit.\n')
        if (inp.strip().lower() == 'q'):
            break
        example = df.sample(1).to_dict('r')[0]
        print(f'Request: {example}')

        response = requests.post(
            url, json=example)
        print(f'Response: {response.json()}')

    print('Exiting.')
