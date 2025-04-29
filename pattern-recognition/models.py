import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image
    transforms.ToTensor() # Convert image to PyTorch tensors for processing
])