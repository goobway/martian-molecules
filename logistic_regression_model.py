import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Define your data paths
DATA_PATH = 'nasa-data/'

# Load the preprocessed data
train_features = pd.read_csv(DATA_PATH + 'preprocessed_train_features.csv', index_col='sample_id')
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv', index_col='sample_id')

# Define the target columns (assuming they are all columns in train_labels)
target_cols = train_labels.columns.tolist()

# Initialize dict to hold fitted models
logreg_model_dict = {}

# Split into binary classifier for each class
for col in target_cols:
    y_train_col = train_labels[col]  # Train on one class at a time

    # Initialize the model
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=2, random_state=42)

    # Fit the model
    logreg_model_dict[col] = clf.fit(train_features.values, y_train_col)

    # Perform Cross-validation
    scores = cross_val_score(clf, train_features.values, y_train_col, cv=5, scoring='neg_log_loss')

    # Output the average log loss for each model
    print(f"Average log loss for {col} model: {-scores.mean()}")  # Multiply by -1 as 'neg_log_loss' is negative

# At this point, 'logreg_model_dict' is a dictionary of trained models, one for each target column.