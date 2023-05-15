import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_sample, int_per_timebin

# Define the base path where the data is located
base_path = 'nasa-data/'

# Load metadata
metadata = pd.read_csv(base_path + "metadata.csv")

# Split metadata into training, validation, and testing
metadata_train = metadata[metadata['split'] == 'train']
metadata_val = metadata[metadata['split'] == 'val']
metadata_test = metadata[metadata['split'] == 'test']

# Get the paths to the first 100 train feature CSV files
first_100_paths = metadata_train['features_path'].iloc[:100]

# List to store the dataframes
dfs = []

# Load each CSV file into a pandas DataFrame and store it in the list
for path in first_100_paths:
  df = pd.read_csv(base_path + path)
  dfs.append(df)

# Load the first 100 training labels
train_labels = pd.read_csv(base_path + "/train_labels.csv", index_col="sample_id")[:100]
target_cols = train_labels.columns

