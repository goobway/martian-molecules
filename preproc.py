import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = 'nasa-data/'

# Read the metadata CSV file
metadata = pd.read_csv(DATA_PATH + 'metadata.csv')
print("Metadata loaded.")

# Filter the metadata to include only train and validation sets
metadata_train = metadata[metadata['split'] == 'train'].head(200)
metadata_val = metadata[metadata['split'] == 'val'].head(50)
metadata = pd.concat([metadata_train, metadata_val])

print("Metadata filtered for first 200 training and first 50 validation samples.")

# Read the labels CSV files
train_labels = pd.read_csv(DATA_PATH + 'train_labels.csv')
val_labels = pd.read_csv(DATA_PATH + 'val_labels.csv')

# Set 'sample_id' as the index for both label dataframes
train_labels.set_index('sample_id', inplace=True)
val_labels.set_index('sample_id', inplace=True)

# Create a dictionary of training files and load training labels
train_files = metadata_train.set_index('sample_id')['features_path'].to_dict()
val_files = metadata_val.set_index('sample_id')['features_path'].to_dict()

# Filter labels for the selected training and validation samples
train_labels = train_labels[train_labels.index.isin(train_files.keys())]
val_labels = val_labels[val_labels.index.isin(val_files.keys())]

# Preprocessing functions
def drop_frac_and_He(df):
    df["rounded_mass"] = df["mass"].transform(round)
    df = df.groupby(["time", "rounded_mass"])["intensity"].aggregate("mean").reset_index()
    df = df[df["rounded_mass"] <= 350]
    df = df[df["rounded_mass"] != 4]
    return df

def remove_background_intensity(df):
    df["intensity_minsub"] = df.groupby(["rounded_mass"])["intensity"].transform(lambda x: (x - x.min()))
    return df

def scale_intensity(df):
    df["int_minsub_scaled"] = MinMaxScaler().fit_transform(df[["intensity_minsub"]])
    return df

def preprocess_sample(df):
    df = drop_frac_and_He(df)
    df = remove_background_intensity(df)
    df = scale_intensity(df)
    return df

# Feature Engineering
timerange = pd.interval_range(start=0, end=25, freq=0.5)
allcombs = list(itertools.product(timerange, [*range(0, 350)]))
allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "rounded_mass"])

def int_per_timebin(df):
    df["time_bin"] = pd.cut(df["time"], bins=timerange)
    df = pd.merge(allcombs_df, df, on=["time_bin", "rounded_mass"], how="left")
    df = df.groupby(["time_bin", "rounded_mass"]).max("int_minsub_scaled").reset_index()
    df = df.replace(np.nan, 0)
    df = df.pivot_table(columns=["rounded_mass", "time_bin"], values=["int_minsub_scaled"])
    return df

# Assembling preprocessed and transformed training set
train_features_dict = {}
print("Total number of train files: ", len(train_files))

for i, (sample_id, filepath) in enumerate(tqdm(train_files.items())):
    temp = pd.read_csv(DATA_PATH + filepath)
    train_sample_pp = preprocess_sample(temp)
    train_sample_fe = int_per_timebin(train_sample_pp).reset_index(drop=True)
    train_features_dict[sample_id] = train_sample_fe

train_features = pd.concat(train_features_dict, names=["sample_id", "dummy_index"]).reset_index(level="dummy_index", drop=True)

assert train_features.index.equals(train_labels.index)

# Save the preprocessed data to a CSV file
train_features.to_csv(DATA_PATH + 'preprocessed_train_features.csv')
