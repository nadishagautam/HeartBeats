import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Step 1: Load DEAM Metadata
# Replace 'metadata.csv' with the actual metadata file from the DEAM dataset
metadata = pd.read_csv("metadata.csv")

# Extract relevant features and labels
features = metadata[["tempo", "energy", "acousticness"]]  # Replace with actual feature columns
labels_valence = metadata["valence"]  # Valence values

# Categorize valence into 3 labels: low, medium, high
def categorize_valence(value):
    if value < 0.33:
        return 0  # Low
    elif value < 0.66:
        return 1  # Medium
    else:
        return 2  # High

metadata["valence_category"] = metadata["valence"].apply(categorize_valence)

# Step 2: Preprocess Data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

labels = metadata["valence_category"].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 3: Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Model parameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 32  # Number of hidden neurons
output_size = 3   # Number of output classes (low, medium, high)

# Initialize the model
model = MLP(input_size, hidden_size, output_size)

# Step 4: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 6: Evaluate the Model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1).numpy()
    y_true = y_test.numpy()

accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\nMLP Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Step 7: Make Predictions on New Songs
new_songs = pd.DataFrame({
    "tempo": [120, 100, 140],
    "energy": [0.8, 0.5, 0.9],
    "acousticness": [0.2, 0.7, 0.1]
})
new_songs_scaled = scaler.transform(new_songs)
new_songs_tensor = torch.tensor(new_songs_scaled, dtype=torch.float32)

with torch.no_grad():
    new_predictions = model(new_songs_tensor)
    new_classes = torch.argmax(new_predictions, axis=1).numpy()
    new_songs["predicted_intensity"] = new_classes

print("\nPredicted Intensity for New Songs:")
print(new_songs)
