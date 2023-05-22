import pandas as pd
import os
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from sklearn.metrics import log_loss, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

torch.manual_seed(42)

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

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Calculate mean and standard deviation of the training data
train_data, train_labels = load_data()
train_mean = train_data.mean(axis=(0, 1, 2))
train_std = train_data.std(axis=(0, 1, 2))

# Define the data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

# Create the dataset
dataset = CustomDataset(train_data, train_labels, transform=transform)

# Create the data loader
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the model
num_classes = 9
model = CNN(num_classes)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "model.pth")

# Load the model
model = CNN(num_classes)
model.load_state_dict(torch.load("model.pth"))

# Make predictions
predictions = []
with torch.no_grad():
    for images, _ in dataloader:
        outputs = model(images)
        predictions.extend(outputs.sigmoid().numpy())

# Assign sample order and target columns from train_labels.csv
labels_df = pd.read_csv(LABELS_PATH)
sample_order = labels_df['sample_id'].tolist()
target_cols = labels_df.columns[1:].tolist()

# Convert predictions to DataFrame
resnet18_df = pd.DataFrame(predictions, index=sample_order, columns=target_cols)
pprint(resnet18_df.head())

# Compute accuracy
y_true = labels_df.iloc[:, 1:].values
y_pred = np.array(predictions) >= 0.5
accuracy = (y_true == y_pred).mean()
print(f"Accuracy: {accuracy:.4f}")

# Save the model
# torch.save(model, "final_model.pt")

# Compute log loss
log_loss_score = log_loss(y_true, resnet18_df.values)
print("Log loss: ", log_loss_score)

# Binarize the labels
val_labels_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

label_names = ['aromatic', 'hydrocarbon', 'carboxylic_acid', 'nitrogen_bearing_compound', 'chlorine_bearing_compound', 'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound', 'mineral']

plt.figure(figsize=(10, 8))

# For each class
for i in range(val_labels_bin.shape[1]):
    precision, recall, _ = precision_recall_curve(val_labels_bin[:, i], resnet18_df.iloc[:, i])
    average_precision = average_precision_score(val_labels_bin[:, i], resnet18_df.iloc[:, i])

    plt.step(recall, precision, where='post', label=f'Precision-Recall curve of {label_names[i]} (area = {average_precision:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
