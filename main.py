# usr/bin/python3
# IMPORTING MODULES AND LIBRARIES
from config import Config as config
from data_loader import CustomTrainDataset
from model import *
from torch.utils.data import DataLoader
from training import train
from validation import validation
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import random_split
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# LOADING DATA
Whole_Dataset = CustomTrainDataset(base_path=config.data_path, subjects=config.subject_to_use,transform=True, validation=False)
train_size = int(config.data_splitting_ratio * len(Whole_Dataset))
test_size = len(Whole_Dataset) - train_size

Train_Dataset, Test_Dataset = random_split(Whole_Dataset, [train_size, test_size])
Train_DataLoader = DataLoader(Train_Dataset, batch_size=config.batch_size, shuffle=False)
Test_DataLoader = DataLoader(Test_Dataset, batch_size=config.batch_size, shuffle=False)

Validation_Dataset = CustomTrainDataset(base_path=config.data_path, subjects=config.subject_to_use,transform=True, validation=True)
Validation_DataLoader = DataLoader(Validation_Dataset, batch_size=config.batch_size, shuffle=False)
print('DataLoader: Success')

# MODEL LOADING
# model = EncoderDecoder(config.input_dim, config.output_dim)
model = Very_Deep_VAE(latent_dim=2, config=config)

# LOADING FUNCTIONS
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = nn.BCELoss()

# TRANIING & TESTING
training_start_time = time.strftime("%c")
train(model, Train_DataLoader, loss_function, optimizer, config)
training_end_time = time.strftime("%c")

# VALIDATION
validation(model, Validation_DataLoader, loss_function, optimizer, config)
