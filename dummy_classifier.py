import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = 'nasa-data/'


def logloss_cross_val(clf, X, y):
    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:
        y_col = y[col]
        log_loss_cv[col] = np.mean(cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer))
    avg_log_loss = np.mean(list(log_loss_cv.values()))
    return log_loss_cv, avg_log_loss


def accuracy_cross_val(clf, X, y):
    # Generate a score for each label class
    accuracy_cv = {}
    for col in y.columns:
        y_col = y[col]
        accuracy_cv[col] = np.mean(cross_val_score(clf, X.values, y_col, cv=skf, scoring='accuracy'))
    avg_accuracy = np.mean(list(accuracy_cv.values()))
    return accuracy_cv, avg_accuracy


# Load train features and labels
train_features = pd.read_csv(DATA_PATH + 'preprocessed_train_features.csv', skiprows=2, index_col='sample_id')
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv', index_col='sample_id')

# Select first 200 samples
train_features = train_features.iloc[:200]
train_labels = train_labels.iloc[:200]

# Check the number of samples
assert train_features.shape[0] == train_labels.shape[0] == 200

# List of target columns
target_cols = train_labels.columns.tolist()

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

# Dummy classifier
dummy_clf = DummyClassifier(strategy="prior")

print("Dummy model cross-validation average log-loss:")
dummy_logloss = logloss_cross_val(dummy_clf, train_features, train_labels[target_cols])
pprint(dummy_logloss[0])
print("\nAggregate log-loss:")
pprint(dummy_logloss[1])
print('')
print("Dummy model cross-validation average accuracy:")
dummy_accuracy = accuracy_cross_val(dummy_clf, train_features, train_labels[target_cols])
pprint(dummy_accuracy[0])
print("\nAggregate accuracy:")
pprint(dummy_accuracy[1])
