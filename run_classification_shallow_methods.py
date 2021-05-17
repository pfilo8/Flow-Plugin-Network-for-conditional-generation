from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils import get_parser_model_flow
from utils.classification import load_dataframes, Y_COLUMN

RANDOM_STATE = 42

results = []

args = get_parser_model_flow().parse_args()
path = args.model_path

df_train, df_test = load_dataframes(path)

print(df_train.shape, df_train.columns)
print(df_test.shape, df_train.columns)

x_train, y_train = df_train.drop(Y_COLUMN, axis=1), df_train[Y_COLUMN]
x_test, y_test = df_test.drop(Y_COLUMN, axis=1), df_test[Y_COLUMN]

models = [
    ('Logistic Regression', partial(LogisticRegression, random_state=RANDOM_STATE), {'C': np.logspace(-2, 2, 5)}),
    ('SVM Linear', partial(SVC, random_state=RANDOM_STATE, kernel='linear'), {'C': np.logspace(-2, 2, 5)}),
    ('SVM RBF', partial(SVC, random_state=RANDOM_STATE, kernel='rbf'), {'C': np.logspace(-2, 2, 5)}),
]

for name, clf, grid in models:
    print(f'Training model {name}.')
    grid = GridSearchCV(
        clf(),
        param_grid=grid,
        n_jobs=-1,
        cv=5,
        verbose=5
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    y_train_hat = model.predict(x_train)
    y_test_hat = model.predict(x_test)

    results.append((name, 'train', accuracy_score(y_train, y_train_hat)))
    results.append((name, 'test', accuracy_score(y_test, y_test_hat)))

print('Saving results.')
results = pd.DataFrame(results)
results.to_csv(path / Path('classification-results-shallow.csv'), index=False)
