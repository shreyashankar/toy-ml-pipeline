from flask import Flask, jsonify, request
from mltrace import get_db_uri, set_db_uri, create_component, tag_component, register, create_random_ids
from prometheus_client import Histogram
from prometheus_flask_exporter import PrometheusMetrics
from utils import io, models

import numpy as np
import pandas as pd

# Create app
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Log static information as metric
metrics.info('app_info', 'Application info', version='0.1')

# Load model
global mw
mw = models.RandomForestModelWrapper.load('training/models', dev=False)
model_path = io.get_output_path('training/models', dev=False)

# Create metrics
buckets = (*(np.around(np.arange(0, 1, 0.05), 2).tolist()), float("inf"))
prediction_histogram = Histogram(name='prediction_histogram', description='Histogram of prediction outputs', registry=metrics.registry, buckets=buckets)

# Prediction function
@app.route('/predict', methods=['POST'])
@register(component_name='inference', inputs=[model_path], input_vars=['feature_path', 'row_idx'], output_vars=['output_ids'], endpoint=True)
def predict():
    req = request.get_json()
    feature_path = req['feature_path'] if 'feature_path' in req else None
    row_idx = f"row_idx_{req['row_idx']}" if 'row_idx' in req else None

    df = pd.read_json(req['data'])
    df['prediction'] = mw.predict(df)
    result = {
        'prediction': df['prediction'].to_list()
    }
    
    # Observe values
    for pred in df['prediction'].to_list():
        prediction_histogram.observe(float(pred))
    
    # Add accuracy if label in df and more than one row
    label_column = 'high_tip_indicator'
    if len(df) > 1 and label_column in df.columns:
        result['score'] = mw.score(df, label_column)


    # Log output ids to mltrace
    output_ids = create_random_ids(num_outputs=len(df))
    result['id'] = output_ids
    out = jsonify(result)
    return out


def main():
    app.run(debug=False, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', '54.177.215.161'))

    # Create components
    create_component(
        'inference', 'API that does inference on a group of features.', 'shreya')
    tag_component('inference', ['inference'])

    main()
