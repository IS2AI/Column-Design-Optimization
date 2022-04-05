import csv
import os
import math
import random
import time
import pandas as pd
import numpy as np
import openseespy.opensees as ops
import openseespy.postprocessing.ops_vis as opsv
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
import cv2

# Define a class with the network architecture
class Net(nn.Module):
    def __init__(self,n_in, n_out, neurons):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_in, 128),
          nn.ReLU(),
          nn.Linear(128, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, neurons),
          nn.ReLU(),
          nn.Linear(neurons, 128),
          nn.Dropout(p=0.1),
          nn.ReLU(),
          nn.Linear(128, n_out)
    )

    def forward(self, x):
        return self.layers(x)

# Load the saved model 
net = torch.load('column.pth')

# Assing the device to create tensors
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu") #torch.device("cpu")

# Allocate the tensor to specified device
net = net.to(device)
loss_func = nn.MSELoss()  

# Set the model in evaluation mode
net.eval()

# Read the file with the normalized test inputs
test_samples = pd.read_hdf("test_samples.h5")
# Convert to torch tensor
X_test = torch.from_numpy(test_samples.to_numpy()).float()

# Extract the network predictions
outp = net(X_test.to(device))

# Transfer to cpu and convering to numpy 
outp = outp.cpu()
out = outp.detach().numpy()

# Creating a dataframe with the outputs 
nn_out = pd.DataFrame(out, columns = ['Width', 'Depth', 'As_total'])

# Read the minimum values of train set
train_min = pd.read_hdf("train_min.h5")
train_min = train_min.drop(['P', 'My', 'Mz', 'fc', 'h'])

# Read the maximum values of train set
train_max = pd.read_hdf("train_max.h5")
train_max = train_max.drop(['P', 'My', 'Mz', 'fc', 'h'])

# Back normalize the test results
back_scaled_nn = train_min + nn_out*(train_max - train_min)

# Save the inference results to csv file
back_scaled_nn.to_csv("nn_combo.csv")
