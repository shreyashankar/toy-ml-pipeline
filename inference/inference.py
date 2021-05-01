"""
inference.py

Sample script that runs inference on examples incrementally. It pings the "api" (served locally in inference/app.py) and gets a response. To use, make sure you are also running inference/app.py.
"""

from utils import io

import argparse
import pandas as pd
import random
import requests
import signal
import sys

parser = argparse.ArgumentParser(description='Run inference.')
parser.add_argument('--start', type=int, help='Start index for inference.', const=0, nargs='?')
args = parser.parse_args()
global idx
idx = args.start if args.start else 0

# Capture index when user quits
def signal_handler(sig, frame):
    print(f'Ending index: {idx}')
    sys.exit(0)

def main():
    # Grab latest features and model wrapper. Sort features by date.
    df = io.load_output_df('features/2020_03')\
           .sort_values(by=['tpep_pickup_datetime'], ascending=True)
    feature_path = io.get_output_path('features/2020_03')

    # Run model on latest features for a random example
    prediction_url = 'http://127.0.0.1:8000/predict'
    log_url = 'http://127.0.0.1:8000/log_prediction'
    batch_size = 256
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    while(True):
        global idx        
        example = df.iloc[range(idx, idx + batch_size)]
        req = {
            'data': example.to_json(),
            'row_idx': idx,
            'feature_path': feature_path
        }

        response = requests.post(
            prediction_url, json=req)
        res = response.json()
        
        result_df = pd.DataFrame({'id': res['id'], 'timestamp': example.tpep_pickup_datetime.to_list(),'prediction': res['prediction'], 'label': example.high_tip_indicator.to_list()})
        result_df['correct'] = (result_df['prediction'] >= 0.5) == (result_df['label'])
        print(result_df)
        print(f"Score: {res['score']}")
        idx += batch_size

    print('Exiting.')


if __name__ == '__main__':
    main()
