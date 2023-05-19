import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATA_PATH = 'nasa-data/2d_images/'
LABELS_PATH = 'nasa-data/train_labels.csv'


def load_data():
    # Get list of numpy files
    npy_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]

    # Load the labels file
    labels_df = pd.read_csv(LABELS_PATH, index_col='sample_id')

    data = []
    labels = []

    for npy_file in npy_files:
        # Load the numpy array
        arr = np.load(DATA_PATH + npy_file)

        # Append to data list
        data.append(arr)

        # Get the corresponding label
        sample_id = npy_file.rstrip('.npy')
        # Fetch the entire row for multiple labels
        label = labels_df.loc[sample_id].values

        # Append to labels list
        labels.append(label)

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def create_model(input_shape=(350, 50, 3), num_classes=9):  # Change 10 to 9
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])

    return model


# Load the data
data, labels = load_data()

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create the model
model = create_model()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=32)
model.summary()
