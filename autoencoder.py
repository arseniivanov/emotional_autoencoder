import numpy as np
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_shape, encoding_dim):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 1024),  
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, np.prod(input_shape)),
            nn.Sigmoid(),
            nn.Unflatten(1, input_shape),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
import torch.nn.functional as F

class AdaptiveAutoencoder(nn.Module):
    def __init__(self, encoding_dim, max_length):
        super(AdaptiveAutoencoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=62, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(64 * (max_length // 4) * 15, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64 * (max_length // 4) * 15),
            nn.ReLU(),
            nn.Unflatten(1, (64, max_length // 4, 15)),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=62, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def save_autoencoder_model(model, path):
    model_info = {
        'input_shape': model.input_shape,
        'encoding_dim': model.encoding_dim,
        'state_dict': model.state_dict()
    }
    torch.save(model_info, path)

def load_autoencoder_model(path):
    model_info = torch.load(path)
    input_shape = model_info['input_shape']
    encoding_dim = model_info['encoding_dim']
    model = Autoencoder(input_shape, encoding_dim)
    model.load_state_dict(model_info['state_dict'])
    return model