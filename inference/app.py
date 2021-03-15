from flask import Flask, jsonify, request
from utils import models

import pandas as pd

app = Flask('high_tip_app')
global mw
mw = models.RandomForestModelWrapper.load('training/models')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    preds = mw.predict(pd.DataFrame({k: [v] for k, v in req.items()}))
    result = {
        'prediction': preds[0]
    }
    return jsonify(result)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
