import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import mne
import scipy.signal
from mne.time_frequency import psd_array_welch
from autoencoder import Autoencoder, save_autoencoder_model, load_autoencoder_model


def map_to_seed_emotions(labels):
    emotions = []
    for valence, arousal, _, _ in labels:
        if 4 <= valence <= 6 and arousal < 4:
            emotions.append(0)  # Neutral
        elif valence < 5 and arousal < 4:
            emotions.append(1)  # Sad
        elif valence < 5 and arousal >= 4:
            emotions.append(2)  # Fear
        elif valence > 5 and arousal >= 4:
            emotions.append(3)  # Happy
        else:
            emotions.append(-1)  # Undefined
    return np.array(emotions)


def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data['data'], data['labels']


class DEAPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def process_deap_data(data, labels):
    # Drop peripheral channels (keeping only the first 32 EEG channels)
    eeg_data = data[:, :32, :]

    # Get the emotion labels
    seed_emotions = map_to_seed_emotions(labels)

    # Filter out the trials with undefined labels (-1)
    valid_indices = np.where(seed_emotions != -1)
    filtered_eeg_data = eeg_data[valid_indices]
    filtered_emotions = seed_emotions[valid_indices]

    return filtered_eeg_data, filtered_emotions 

def preprocess_deap_data(eeg_data):
    # Create a Raw object from the given EEG data
    ch_names = [f'EEG {i+1}' for i in range(32)]
    ch_types = ['eeg'] * 32
    sfreq = 128
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(eeg_data.T, info)

    # Apply bandpass filtering
    raw.filter(l_freq=4, h_freq=45, method='fir', fir_design='firwin')
    filtered_data = raw.get_data().T
    # Calculate the PSD using Welch's method
    psd = psd_array_welch(filtered_data, fmin=4, fmax=45, n_fft=64, n_overlap=32, verbose=False, sfreq=sfreq)

    # Log-transform the PSD values to get the approximate DE
    de = np.log(psd)

    # Average the PSD and DE values across all trials
    avg_psd = np.mean(psd, axis=0)
    avg_de = np.mean(de, axis=0)

    return np.concatenate([avg_psd, avg_de], axis=-1)

def prepare_deap_data_for_pytorch(data, labels):
    # Process the data to keep only EEG channels and valid emotion labels
    eeg_data, emotions = process_deap_data(data, labels)

    # Preprocess the EEG data using the preprocess_deap_data function
    preprocessed_data = []
    for trial_data in eeg_data:
        trial_preprocessed = preprocess_deap_data(trial_data[:, :32])
        preprocessed_data.append(trial_preprocessed)
    preprocessed_data = np.array(preprocessed_data)

    # Split the data into train and test sets (80/20)
    train_data, test_data, train_labels, test_labels = train_test_split(
        preprocessed_data, emotions, test_size=0.2, random_state=42)

    # Create PyTorch Datasets and DataLoaders
    train_dataset = DEAPDataset(train_data, train_labels)
    test_dataset = DEAPDataset(test_data, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, test_dataloader

# Load data and labels for a specific participant (e.g., s01.dat)
data, labels = load_deap_data('DEAP_data/s01.dat')
seed_emotions = map_to_seed_emotions(labels)

train_dataloader, test_dataloader = prepare_deap_data_for_pytorch(data, labels)

# Load the state_dict
autoencoder_model = load_autoencoder_model('autoencoder_model.pth')

# Set the model to evaluation mode
autoencoder_model.eval()
# Evaluate the model on DEAP test data
for inputs, labels in test_dataloader:
    import pdb
    pdb.set_trace()
    inputs = inputs.float()
    outputs = autoencoder_model(inputs)
