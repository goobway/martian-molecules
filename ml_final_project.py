import itertools
from pathlib import Path
from pprint import pprint
import os
import shutil

from matplotlib import pyplot as plt, cm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

from preprocessing import preprocess_sample, drop_frac_and_He, remove_background_intensity

pd.set_option("max_colwidth", 80)
RANDOM_SEED = 42  # For reproducibility
tqdm.pandas()

# from google.colab import drive
# drive.mount('/content/gdrive')

PROJ_ROOT = Path.cwd().parent
DATA_PATH = "nasa-data"
metadata = pd.read_csv(DATA_PATH + "/metadata.csv",
                       index_col="sample_id")[:200]
metadata.head()

train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))

sample_ids_ls = metadata.index.values[0:5]

# Import datasets for EDA
sample_data_dict = {}

for sample_id in sample_ids_ls:
    sample_data_dict[sample_id] = pd.read_csv(
        DATA_PATH + "/" + train_files[sample_id])


def plot_spectrogram(sample_df, sample_lab):

    # For visual clarity, we will round these intensity values to the nearest whole number and average the intensity.
    sample_df["mass"] = sample_df["mass"].round()
    sample_df = (
        sample_df.groupby(["time", "mass"])[
            "intensity"].aggregate("mean").reset_index()
    )

    masses = sample_df["mass"].to_numpy()
    times = sample_df["time"].to_numpy()
    intensities = sample_df["intensity"].to_numpy()

    for m in np.unique(masses):
        mask = masses == m
        plt.plot(times[mask], intensities[mask])

    plt.title(sample_lab)


fig, ax = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
fig.suptitle("Samples")
fig.supxlabel("Time")
fig.supylabel("Intensity")

for i in range(0, 5):
    sample_lab = sample_ids_ls[i]
    sample_df = sample_data_dict[sample_lab]

    plt.subplot(1, 5, i + 1)
    plot_spectrogram(sample_df, sample_lab)

train_labels = pd.read_csv(
    DATA_PATH + "/train_labels.csv", index_col="sample_id")[:200]
target_cols = train_labels.columns
train_labels.head()

train_labels.aggregate("sum", axis=1).value_counts(normalize=True)

sumlabs = train_labels.aggregate("sum").sort_values()

plt.barh(sumlabs.index, sumlabs, align="center")
plt.ylabel("Compounds")
plt.xticks(rotation=45)
plt.xlabel("Count in training set")
plt.title("Compounds represented in training set")
plt.show()

sample_df = pd.read_csv(
    DATA_PATH + "/" + metadata[metadata.split == "train"].features_path.iloc[0]
)
sample_df

# Calculate summary statistics for time and m/z values for training set


def get_time_mass_stats(fpath):

    df = pd.read_csv(DATA_PATH + "/" + fpath)

    time_min = df["time"].min()
    time_max = df["time"].max()
    time_range = time_max - time_min

    mass_min = df["mass"].min()
    mass_max = df["mass"].max()
    mass_range = mass_max - mass_min

    return time_min, time_max, time_range, mass_min, mass_max, mass_range


sample_paths_ls = metadata[metadata["split"] == "train"].features_path[1:]
training_stats_df = pd.DataFrame(sample_paths_ls)
training_stats_df.columns = ["fpath"]

(
    training_stats_df["time_min"],
    training_stats_df["time_max"],
    training_stats_df["time_range"],
    training_stats_df["mass_min"],
    training_stats_df["mass_max"],
    training_stats_df["mass_range"],
) = zip(*training_stats_df["fpath"].progress_apply(get_time_mass_stats))

training_stats_df.describe()

sample_df = drop_frac_and_He(sample_df)

# Intensity values before subtracting minimum
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Intensity values across time by m/z")
fig.supxlabel("Time")
fig.supylabel("Intensity")

plt.subplot(1, 2, 1)

rounded_masses = sample_df["rounded_mass"].to_numpy()
times = sample_df["time"].to_numpy()
intensities = sample_df["intensity"].to_numpy()

for m in np.unique(rounded_masses):
    mask = rounded_masses == m
    plt.plot(times[mask], intensities[mask])

plt.title("Before subtracting minimum intensity")

# After subtracting minimum intensity value
sample_df = remove_background_intensity(sample_df)

plt.subplot(1, 2, 2)

rounded_masses = sample_df["rounded_mass"].to_numpy()
times = sample_df["time"].to_numpy()
intensity_minsubs = sample_df["intensity_minsub"].to_numpy()

for m in np.unique(rounded_masses):
    mask = rounded_masses == m
    plt.plot(times[mask], intensity_minsubs[mask])

plt.title("After subtracting minimum intensity")
plt.show()


fig, ax = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
fig.suptitle("Training samples")
fig.supxlabel("Time (m)")
fig.supylabel("Relative Intensity")

for i in range(0, 5):
    sample_lab = sample_ids_ls[i]
    sample_df = sample_data_dict[sample_lab]
    sample_df = preprocess_sample(sample_df)

    plt.subplot(1, 5, i + 1)

    rounded_masses = sample_df["rounded_mass"].to_numpy()
    times = sample_df["time"].to_numpy()
    int_minsub_scaled = sample_df["int_minsub_scaled"].to_numpy()

    for m in np.unique(rounded_masses):
        mask = rounded_masses == m
        plt.plot(times[mask], int_minsub_scaled[mask])

    plt.title(sample_lab)

# Create a series of time bins
timerange = pd.interval_range(start=0, end=25, freq=0.5)
timerange

# Make dataframe with rows that are combinations of all temperature bins and all m/z values
allcombs = list(itertools.product(timerange, [*range(0, 350)]))

allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "rounded_mass"])
allcombs_df.head()


def int_per_timebin(df):
    """
    Transforms dataset to take the preprocessed max abundance for each
    time range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """

    # Bin times
    df["time_bin"] = pd.cut(df["time"], bins=timerange)

    # Combine with a list of all time bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["time_bin", "rounded_mass"], how="left")

    # Aggregate to time bin level to find max
    df = df.groupby(["time_bin", "rounded_mass"]).max(
        "int_minsub_scaled").reset_index()

    # Fill in 0 for intensity values without information
    df = df.replace(np.nan, 0)

    # Reshape so each row is a single sample
    df = df.pivot_table(
        columns=["rounded_mass", "time_bin"], values=["int_minsub_scaled"]
    )

    return df

# NEW -- function to save preprocessed data
def save_transformed_data(df, output_file):
    """
    Saves the transformed dataframe after time binning into a new CSV file.

    Args:
        df: Transformed dataframe after time binning.
        output_file: File path to save the transformed data.

    Returns:
        None
    """
    df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}.")

# Assembling preprocessed and transformed training set


train_features_dict = {}
print("Total number of train files: ", len(train_files))

for i, (sample_id, filepath) in enumerate(tqdm(train_files.items())):

    # Load training sample
    temp = pd.read_csv(DATA_PATH + "/" + filepath)

    # Preprocessing training sample
    train_sample_pp = preprocess_sample(temp)

    # Feature engineering
    train_sample_fe = int_per_timebin(train_sample_pp).reset_index(drop=True)
    train_features_dict[sample_id] = train_sample_fe

    output_file = f"nasa-data/preproc_features/S{i}.csv"
    save_transformed_data(train_sample_fe, output_file)

train_features = pd.concat(
    train_features_dict, names=["sample_id", "dummy_index"]
).reset_index(level="dummy_index", drop=True)

# Make sure that all sample IDs in features and labels are identical
assert train_features.index.equals(train_labels.index)

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

# Check log loss score for baseline dummy model


def logloss_cross_val(clf, X, y):

    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:

        y_col = y[col]  # take one label at a time
        log_loss_cv[col] = np.mean(
            cross_val_score(clf, X.values, y_col, cv=skf,
                            scoring=log_loss_scorer)
        )

    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss


# Dummy classifier
dummy_clf = DummyClassifier(strategy="prior")

print("Dummy model cross-validation average log-loss:")

dummy_logloss = logloss_cross_val(
    dummy_clf, train_features, train_labels[target_cols])
pprint(dummy_logloss[0])
print("\nAggregate log-loss")
print(dummy_logloss[1])

# Logression Classifier
logreg_clf = LogisticRegression(penalty="l1", solver="liblinear", C=2.0)
print("Logistic regression model cross-validation average log-loss:\n")
logreg_logloss = logloss_cross_val(
    logreg_clf, train_features, train_labels[target_cols])
pprint(logreg_logloss[0])
print("Aggregate log-loss")
print(logreg_logloss[1])


def row2img(row, save=False):
    # print(row.values.shape)
    temp_arr = row.values.reshape(350, 50, 1)

    # Scale from 0-1 to 0-255, make uint8, squeeze to just two dims
    temp_arr = np.squeeze((temp_arr * 255).astype(np.uint8), axis=2)

    # Stack to 3 dims to mimic rgb image
    temp_arr = np.stack((temp_arr,) * 3, axis=-1)
    temp_img = Image.fromarray(temp_arr)

    # Save as image
    if save:
        IMG_OUTPUT_PATH = 'images/'
        outpath = IMG_OUTPUT_PATH / (
            str(metadata.loc[row.name]["features_path"])[0:-4] + ".jpeg"
        )
        temp_img.save(outpath)

    return temp_img


fig, axes = plt.subplots(4, 2, figsize=(20, 10))
for i in range(0, 4):
    # print(train_features)
    sample_lab = sample_ids_ls[i]
    sample_df = sample_data_dict[sample_lab]
    sample_df = preprocess_sample(sample_df)
    rounded_masses = sample_df["rounded_mass"].to_numpy()
    times = sample_df["time"].to_numpy()
    int_minsub_scaled = sample_df["int_minsub_scaled"].to_numpy()

    for i, m in enumerate(np.sort(np.unique(rounded_masses))):
        mask = rounded_masses == m
        axes[i % 4][0].plot(times[mask], int_minsub_scaled[mask])

    # train_sample_img = row2img(train_features.iloc[i])
    # axes[i % 4][1].imshow(train_sample_img, aspect='auto')
plt.show()
