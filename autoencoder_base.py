import numpy as np
import pandas as pd
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from autoencoder import Autoencoder, AdaptiveAutoencoder, save_autoencoder_model, load_autoencoder_model

class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        else:
            self.best_score = val_loss
            self.counter = 0

def load_seed_iv_data(data_path, session):
    data_path = os.path.join(data_path, 'eeg_feature_smooth', str(session))
    
    subject_data = []
    subject_labels = []  # Initialize subject_labels here
    session_label_dict = {
        1 : [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        2 : [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        3 : [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    }

    for file_name in sorted(os.listdir(data_path)):
        if file_name.endswith('.mat'):
            file_path = os.path.join(data_path, file_name)
            data = sio.loadmat(file_path)
            
            # Load labels for this subject
            labels = session_label_dict[session]
            
            trials_data = []
            for trial in range(1, 25):
                trial_data = []
                for feature_type in ['psd', 'de']:
                    for smoothing_method in ['movingAve', 'LDS']:
                        trial_data.append(data[f'{feature_type}_{smoothing_method}{trial}'])
                
                trials_data.append(trial_data)
            
            subject_data.append(trials_data)
            subject_labels.append(np.array(labels))
    
    return subject_data, subject_labels


data_path = 'SEED_IV_data'
session = 1

subject_data, subject_labels = load_seed_iv_data(data_path, session)

# Find the maximum length of W
max_length = 0
for subject in subject_data:
    for trial in subject:
        if trial[0].shape[1] > max_length:
            max_length = trial[0].shape[1]

# Pad the trials to the same length and concatenate feature channels
subject_data_padded = []
for subject in subject_data:
    subject_trials_padded = []
    for trial in subject:
        trial_padded = np.zeros((62, max_length, 20))
        trial_padded[:, :trial[0].shape[1], :] = np.concatenate(trial, axis=-1)
        subject_trials_padded.append(trial_padded)
    subject_data_padded.append(np.stack(subject_trials_padded))

# Combine data from all subjects
all_data = np.concatenate(subject_data_padded, axis=0)
all_labels = np.concatenate(subject_labels, axis=0)

# Normalize the data
all_data_normalized = (all_data - all_data.mean(axis=(1, 2), keepdims=True)) / all_data.std(axis=(1, 2), keepdims=True)

# Create subject groups
subjects = np.repeat(np.arange(len(subject_data_padded)), 24)

# Split the data using GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(gss.split(all_data_normalized, all_labels, subjects))

# Create train and test datasets
X_train, X_test = all_data_normalized[train_indices], all_data_normalized[test_indices]
y_train, y_test = all_labels[train_indices], all_labels[test_indices]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert data to PyTorch tensors and move them to the device (GPU or CPU)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
test_dataset = TensorDataset(X_test_tensor, X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(EmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        return out


# Create the autoencoder
input_shape = X_train.shape[1:]
encoding_dim = 128
autoencoder = Autoencoder(input_shape, encoding_dim).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(autoencoder.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_fn = nn.MSELoss()
early_stopping = EarlyStopping(patience=5)

# Train the autoencoder
num_epochs = 100

for epoch in range(num_epochs):
    for batch_X, _ in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(batch_X)

        # Calculate the loss
        loss = loss_fn(outputs, batch_X)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Test the autoencoder
    test_loss = 0
    autoencoder.eval()
    with torch.no_grad():
        for batch_X, _ in test_loader:
            test_X = batch_X.to(device)
            test_output = autoencoder(test_X)
            test_loss += loss_fn(test_output, test_X).item()
    
    test_loss /= len(test_loader)
    scheduler.step(test_loss)
    early_stopping(test_loss)
    if early_stopping.early_stop:
        break

    autoencoder.train()

    # Print the loss for this epoch
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {test_loss}')

# Freeze autoencoder and train embedding-predictor
for param in autoencoder.encoder.parameters():
    param.requires_grad = False

# Convert labels to PyTorch tensors and move them to the device (GPU or CPU)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create TensorDataset and DataLoader for training LSTM
train_dataset_lstm = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset_lstm = TensorDataset(X_test_tensor, y_test_tensor)

train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=32, shuffle=True)
test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=32, shuffle=False)

# Define the LSTM model
input_dim = encoding_dim
hidden_dim = 64
num_layers = 1
output_dim = len(np.unique(all_labels))  # Number of unique emotion labels
lstm_model = EmotionClassifier(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Define the optimizer and loss function for LSTM
optimizer_lstm = torch.optim.Adam(lstm_model.parameters())
loss_fn_lstm = nn.CrossEntropyLoss()

# Train the LSTM model
num_epochs_lstm = 100
for epoch in range(num_epochs_lstm):
    for batch_X, batch_y in train_loader_lstm:
        # Zero the gradients
        optimizer_lstm.zero_grad()

        # Generate embeddings using the encoder
        embeddings = autoencoder.encoder(batch_X).unsqueeze(1)

        # Forward pass
        outputs_lstm = lstm_model(embeddings)

        # Calculate the loss
        loss_lstm = loss_fn_lstm(outputs_lstm, batch_y)

        # Backward pass
        loss_lstm.backward()

        # Update the weights
        optimizer_lstm.step()

    # Print the loss for this epoch
    print(f'Epoch: {epoch+1}/{num_epochs_lstm}, Loss: {loss_lstm.item()}')

# Save the autoencoder and LSTM models
torch.save(autoencoder.state_dict(), "autoencoder_model.pth")
torch.save(lstm_model.state_dict(), "lstm_model.pth")

# Evaluate the model on the test dataset
lstm_model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader_lstm:
        # Generate embeddings using the encoder
        embeddings = autoencoder.encoder(batch_X).unsqueeze(1)

        # Forward pass
        outputs_lstm = lstm_model(embeddings)

        # Get the predictions
        _, predicted = torch.max(outputs_lstm, 1)

        # Store the true and predicted labels
        y_true.extend(batch_y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
