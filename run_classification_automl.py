from pathlib import Path

import h2o
import pandas as pd

from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score

from utils import get_parser_model_flow
from utils.classification import load_dataframes, Y_COLUMN

RANDOM_STATE = 42

results = []

h2o.init()

args = get_parser_model_flow().parse_args()
path = args.model_path

df_train, df_test = load_dataframes(path)

print(df_train.shape, df_train.columns)
print(df_test.shape, df_train.columns)

dataset_train = h2o.H2OFrame(df_train)
dataset_test = h2o.H2OFrame(df_test)

# Identify predictors and response
x = dataset_train.columns
x.remove(Y_COLUMN)

# For binary classification, response should be a factor
dataset_train[Y_COLUMN] = dataset_train[Y_COLUMN].asfactor()
dataset_test[Y_COLUMN] = dataset_test[Y_COLUMN].asfactor()

# Run AutoML 
aml = H2OAutoML(max_runtime_secs=60 * 60, seed=RANDOM_STATE)
aml.train(x=x, y=Y_COLUMN, training_frame=dataset_train)

lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

y_train = dataset_train[Y_COLUMN].as_data_frame()
y_train_hat = aml.predict(dataset_train).as_data_frame()['predict']
y_test = dataset_test[Y_COLUMN].as_data_frame()
y_test_hat = aml.predict(dataset_test).as_data_frame()['predict']

results.append(('AutoML', 'train', accuracy_score(y_train, y_train_hat)))
results.append(('AutoML', 'test', accuracy_score(y_test, y_test_hat)))

print('Saving results.')
results = pd.DataFrame(results)
results.to_csv(path / Path('classification-automl.csv'), index=False)
