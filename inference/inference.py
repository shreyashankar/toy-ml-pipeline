import pandas as pd

from utils import io, models

if __name__ == '__main__':
    # Grab latest features and model wrapper
    df = io.load_output_df('features')
    mw = models.RandomForestModelWrapper.load('training/models')

    # Run model on latest features
    preds = mw.predict(df)

    # Dump outputs
    preds_df = pd.DataFrame({'prediction': preds})
    print(io.save_output_df(preds_df, 'inference'))
