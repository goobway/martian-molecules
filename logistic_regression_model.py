import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize

DATA_PATH = 'nasa-data/'

# Load train features and labels
train_features = pd.read_csv(DATA_PATH + 'preprocessed_train_features.csv', skiprows=2, index_col='sample_id')
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv', index_col='sample_id')

# Check the number of samples
assert train_features.shape[0] == train_labels.shape[0] == 809

# List of target columns
target_cols = train_labels.columns.tolist()

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=5, random_state=44, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

# Define the logistic regression model with L1 regularization
logreg_clf = LogisticRegression(penalty="l1", solver="liblinear", C=2)

# Define the function for cross-validation and log-loss calculation
def logloss_cross_val(clf, X, y):
    # Initialize a dictionary to store the log-loss for each label class
    log_loss_cv = {}
    
    # Loop over each label class
    for col in y.columns:
        # Get the labels for the current class
        y_col = y[col]

        # Calculate the cross-validation score, which is the average log-loss over all folds
        log_loss_cv[col] = np.mean(cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer))

    # Calculate the aggregate log-loss, which is the average log-loss over all label classes
    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss

def accuracy_cross_val(clf, X, y):
    # Generate a score for each label class
    accuracy_cv = {}
    for col in y.columns:
        y_col = y[col]  # take one label at a time
        accuracy_cv[col] = np.mean(
            cross_val_score(clf, X.values, y_col, cv=skf, scoring='accuracy')
        )
    avg_accuracy = np.mean(list(accuracy_cv.values()))
    return accuracy_cv, avg_accuracy

# Print the cross-validation log-loss for the logistic regression model
print("Logistic regression model cross-validation average log-loss:")
logreg_logloss = logloss_cross_val(logreg_clf, train_features, train_labels[target_cols])
pprint(logreg_logloss[0])
print("\nAggregate log-loss:")
pprint(logreg_logloss[1])
print('')
print("Logistic regression model cross-validation average accuracy:")
logreg_accuracy = accuracy_cross_val(logreg_clf, train_features, train_labels[target_cols])
pprint(logreg_accuracy[0])
print("\nAggregate accuracy:")
pprint(logreg_accuracy[1])

# Define the function to train the logistic regression model using all of the available training data
def logreg_train(X_train, y_train):
    # Initialize a dictionary to store the trained models for each label class
    logreg_model_dict = {}

    # Loop over each label class
    for col in y_train.columns:
        # Get the labels for the current class
        y_train_col = y_train[col]

        # Initialize a new logistic regression model
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=2, random_state=42)

        # Train the model and store it in the dictionary
        logreg_model_dict[col] = clf.fit(X_train.values, y_train_col)

    return logreg_model_dict

# Train the logistic regression model using all of the available training data
fitted_logreg_dict = logreg_train(train_features, train_labels[target_cols])

# binarize the labels
y_bin = label_binarize(train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# Initialize a dictionary to store the trained models for each label class
logreg_model_dict = {}

plt.figure(figsize=(10, 8))

# Loop over each label class
for i, col in enumerate(train_labels.columns):
    # Get the labels for the current class
    y_train_col = y_bin[:, i]

    # Initialize a new logistic regression model
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=2, random_state=44)

    # Train the model and store it in the dictionary
    clf.fit(train_features.values, y_train_col)

    # Predict the probabilities for the current class
    y_score = clf.decision_function(train_features.values)

    # Compute Precision-Recall and plot curve
    precision, recall, _ = precision_recall_curve(y_train_col, y_score)
    average_precision = average_precision_score(y_train_col, y_score)

    plt.step(recall, precision, where='post', label=f'Precision-Recall curve of class {col} (area = {average_precision:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
