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
import cv2
import time
from sklearn.metrics import mean_squared_error
start_time = time.time()

# Loading train, validation and test sets
train = pd.read_hdf('train_norm.h5')
validation = pd.read_hdf('val_norm.h5')
test = pd.read_hdf('test_norm.h5')

# Splitting to input X and target y sets
X_train = train[['P', 'My', 'Mz', 'fc', 'h']]
y_train = train[[ 'Width', 'Depth','As_total']]

X_val = validation[['P', 'My', 'Mz', 'fc', 'h']]
y_val = validation[[ 'Width', 'Depth','As_total']]

X_test = test[['P',  'My', 'Mz', 'fc', 'h']]
y_test = test[['Width', 'Depth','As_total']]

### Converting to torch Tensor format (for torch trainig)
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy()).float()

X_val = torch.from_numpy(X_val.to_numpy()).float()
y_val = torch.from_numpy(y_val.to_numpy()).float()

X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.from_numpy(y_test.to_numpy()).float()

# Print the shape of tensors
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Loading data to Data Loader to iterate over
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,      
    batch_size=batch_size,  
    shuffle = True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,      
    batch_size=batch_size,      
)


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

# Create an object of class Net
net = Net(X_train.shape[1], y_train.shape[1], 256)
print(net)

# Define the network hyperparameters
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Define the device to train the network
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu") #torch.device("cpu")
print(device)

# Allocate the tensor to specified device
net = net.to(device)

# Define the loss function
loss_func = nn.MSELoss()  

# Initialize a dictionary to store training and validation losses
H = {"train_loss": [], "val_loss": []}

# Define number of epochs
epochs=150

# Calculate total train and validation steps
trainSteps = len(train_loader.dataset) // batch_size
valSteps = len(val_loader.dataset) // batch_size

# Iterate over the number of epochs
for epoch in range(epochs):
    # Set the model in training mode
    net.train()
    
    # Initialize the total training and validation losses
    totalTrainLoss = 0
    totalValLoss = 0
    total_loss = 0
    
    # Loop over the training set
    for i,batch in enumerate(train_loader):
        x_train,y_target = batch
        
        # Send the inputs and outputs to the device
        x_train, y_target = x_train.to(device),y_target.to(device)
        
        y_pred = net(x_train)
        
        # Compute the loss
        train_loss = loss_func(y_pred, y_target)
    
        optimizer.zero_grad()                  # Clear gradients for next train
        train_loss.backward()                  # Backpropagation, compute gradients
        optimizer.step()                       # Apply gradients
        
        totalTrainLoss += train_loss           # Cumulative train loss
        
    print("Epoch # %i, train_loss=%f"%(epoch, totalTrainLoss))

    with torch.no_grad():
        # Set the model in evaluation mode
        net.eval()
        for i,batch in enumerate(val_loader):
            x_val,y_val = batch
            
            # Send the inputs and outputs to the device
            x_val, y_val = x_val.to(device),y_val.to(device)
            pred = net(x_val)
            
            # Cumulative validation loss
            totalValLoss += loss_func(pred, y_val)
    
    # Calculate the average training and validation losses
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # Update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())

# Saving the model
MODEL_PATH = 'fc_30.pth' 
torch.save(net, MODEL_PATH)

# Plotting the training and validation accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Training and validation lossess")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(str(batch_size)+'_lr= '+str(learning_rate)+'_dp=0.3'+'_.png')

# Set the model in evaluation mode
net.eval()

# Pass test test through the network
outp = net(X_test.to(device))

# Move the tensors to the cpu
outp = outp.cpu()
out = outp.detach().numpy()

# Create a dataframe from the network test outputs
nn_out = pd.DataFrame(out, columns = ['Width', 'Depth', 'As_total'])

# Extract the minimum values from the training set
train_min = pd.read_hdf("train_min.h5")
train_min = train_min.drop(['P', 'My', 'Mz', 'fc', 'h'])

# Extract the maximum values from the training set
train_max = pd.read_hdf("train_max.h5")
train_max = train_max.drop(['P', 'My', 'Mz', 'fc', 'h'])

# Load the groundtruth values for the test set
test_orig = pd.read_hdf("test.h5")
test_orig = test_orig.drop(columns=['P','My', 'Mz', 'fc', 'h'])

# Back normalize the network output based on the min and max values of training set
back_scaled_nn = train_min + nn_out*(train_max - train_min) 

# Compute MSE loss for the back normalized values
error_scaled = mean_squared_error(back_scaled_nn, test_orig, multioutput='raw_values')

# Compute MSE loss for normalized values
error = mean_squared_error(nn_out,y_test, multioutput='raw_values')

print(error_scaled)
print(error)




