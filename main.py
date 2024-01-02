# usr/bin/python3
# IMPORTING MODULES AND LIBRARIES
from config import Config as config
from data_loader import CustomTrainDataset
from model import *
from torch.utils.data import DataLoader
from training import train
from torch.optim import Adam
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# LOADING DATA
data_loader = CustomTrainDataset(base_path=config.data_path, subjects=config.subject_to_use,transform=Truek, Train=False)
train_loader = DataLoader(data_loader, batch_size=config.batch_size, shuffle=False)

data_loader = CustomTrainDataset(base_path=config.data_path, subjects=config.subject_to_use, transform=True, Train=False)
test_loader = DataLoader(data_loader, )
print('DataLoader: Success')


# LOADING MODEL AND OPTIMIZER AND LOSS FUNCTION
# model = EncoderDecoder(config.input_dim, config.output_dim)
model = Very_Deep_VAE(latent_dim=2, config=config)

optimizer = Adam(model.parameters(), lr=0.001)
loss_function = Very_Deep_VAE.loss_function
train(model, train_loader, loss_function, optimizer, config)