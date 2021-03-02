import pandas as pd

from utils import io

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [2, 4, 6]})
print(io.save_output(df, 'clean', version='test'))

# print(io.list_files('dev/clean/test.pq'))
