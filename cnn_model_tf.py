import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Conv1D, Reshape
from sklearn.metrics import precision_recall_curve, average_precision_score,log_loss
from sklearn.preprocessing import label_binarize


DATA_PATH = 'nasa-data/2d_images/'
LABELS_PATH = 'nasa-data/train_labels.csv'


def load_data():
    npy_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
    labels_df = pd.read_csv(LABELS_PATH, index_col='sample_id')

    data = []
    labels = []

    for npy_file in npy_files:
        arr = np.load(DATA_PATH + npy_file)
        data.append(arr)
        sample_id = npy_file.rstrip('.npy')
        label = labels_df.loc[sample_id].values
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def create_model(input_shape=(350, 50, 3), num_classes=9):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        Reshape((-1, 50, 32)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])

    return model


data, labels = load_data()
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=32)
model.summary()

# Predict the probabilities of each class
pred_probs = model.predict(val_data)

# Compute log loss
log_loss_score = log_loss(val_labels, pred_probs)
print("Log loss: ", log_loss_score)

# Binarize the labels
val_labels_bin = label_binarize(val_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

label_names = ['aromatic', 'hydrocarbon', 'carboxylic_acid', 'nitrogen_bearing_compound', 'chlorine_bearing_compound', 'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound', 'mineral']

plt.figure(figsize=(10, 8))

# For each class
for i in range(val_labels_bin.shape[1]):
    precision, recall, _ = precision_recall_curve(val_labels_bin[:, i], pred_probs[:, i])
    average_precision = average_precision_score(val_labels_bin[:, i], pred_probs[:, i])

    plt.step(recall, precision, where='post', label=f'Precision-Recall curve of {label_names[i]} (area = {average_precision:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
