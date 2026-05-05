#Imports
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_setup import create_dataset

#Paths
monet_path = 'data/monet_jpg'
photo_path = 'data/photo_jpg'

#Hyperparameters
BATCH_SIZE = 32

#Create dataloaders
monet_loader, photo_loader = create_dataset(monet_path=monet_path, photo_path=photo_path, batch_size=BATCH_SIZE)

