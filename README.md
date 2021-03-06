# Column-Design-Optimization

## Requirements:
1. OS Windows/Linux/Mac.
2. Anaconda
3. Python>3.8


## How to use?

### 1: Setting working environment
Create an environment using the env.yaml in Anaconda with the following command:

conda env create -f env.yaml

### 2: Data generation
The data generation folder contains files for parametric data generation.

- main.py is the main script to run the data generation. 

- column.py script contrains the material, geometric, analysis, and model parameters.

- functions.py contains function used to generate data, such as section analysis, random generation of section geometry.

- data.csv file contains the sample output from data generation.

### 3: Data pre-processing and network model
- pre-processing.py script contains the data filtration based on monetary cost for the case of 4.0 meters.

- normalization.py script contrains min-max normalization and preparing the data for network.

- net.py file contrains network file for training the network. 

- use column.pth  and test.py files for inference.

- use test_min.h5 and test_max.h5 files for minmax normalization of the test sample before extracting network predictions.

### 4: Design check
Use check.py to run the check of the network output results.

