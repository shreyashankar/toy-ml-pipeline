from flask import Flask, jsonify, request
from mltrace import get_db_uri, set_db_uri, create_component, tag_component, register, create_random_ids
from prometheus_flask_exporter import PrometheusMetrics
from utils import io, models

import pandas as pd

app = Flask('high_tip_app')
metrics = PrometheusMetrics(app)

# static information as metric
metrics.info('app_info', 'Application info', version='0.1')

global mw
mw = models.RandomForestModelWrapper.load('training/models')
model_path = io.get_output_path('training/models')


def log_live_metric(name: str, val: float):
    pass

@register(component_name='inference', inputs=[model_path], input_vars=['feature_path', 'row_idx'], output_vars=['output_ids'])
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    feature_path = req['feature_path'] if 'feature_path' in req else None
    row_idx = req['row_idx'] if 'row_idx' in req else None
    df = pd.DataFrame({k: [v] for k, v in req['data'].items()})
    df['prediction'] = mw.predict(df)
    result = {
        'prediction': df['prediction']
    }
    # Add accuracy if label in df and more than one row
    label_column = 'high_tip_indicator'
    if len(df) > 1 and label_column in df.columns:
        result['score'] = mw.score(df, label_column)

    # Log output ids to mltrace
    output_ids = create_random_ids(num_outputs=len(df))
    return jsonify(result)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    set_db_uri(get_db_uri().replace('database', 'localhost'))

    # Create components
    create_component(
        'inference', 'API that does inference on a group of features.', 'shreya')
    tag_component('inference', ['inference'])

    main()
