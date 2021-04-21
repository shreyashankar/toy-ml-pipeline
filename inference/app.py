from flask import Flask, jsonify, request
from utils import models

import pandas as pd

app = Flask('high_tip_app')
global mw
mw = models.RandomForestModelWrapper.load('training/models')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    df = pd.DataFrame({k: [v] for k, v in req.items()})
    df['prediction'] = mw.predict(df)
    result = {
        'prediction': df['prediction']
    }
    # Add accuracy if label in df and more than one row
    label_column = 'high_tip_indicator'
    if df.count() > 1 and label_column in df.columns:
        result['score'] = mw.score(df, label_column)
    return jsonify(result)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
