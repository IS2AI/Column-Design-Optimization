import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Load the pre-processsed dataset
df = pd.read_hdf('pre-processed.h5')

# Shuffling the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Splitting the first 85% of the dataset for train set 
train = df[:round(len(df)*0.85)]
# Min-max normalization of the train set
train_norm = (train - train.min())/(train.max() - train.min())
train_min = train.min()
train_max = train.max()
# Save the train set
train_norm.to_hdf("train_norm.h5", key='w')

# Save train set min and max values for back normalization of the inference cases
train_min.to_hdf("train_min.h5", key='w')
train_max.to_hdf("train_max.h5", key='w')

# Splitting and normalizing the validation set
val = df[round(len(df)*0.85):round(len(df)*0.95)]
val.to_hdf("val.h5", key='w')
val_norm = (val - train.min()) / (train.max() - train.min())
# Save the validation set
val_norm.to_hdf("val_norm.h5", key='w')

# Splitting and normalizing the validation set
test = df[round(len(df)*0.95):]
test.to_hdf("test.h5", key='w')
test_norm = (test - train.min()) / (train.max() - train.min())
# Save the test set
test_norm.to_hdf("test_norm.h5", key='w')





