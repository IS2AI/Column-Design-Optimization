# Column-Design-Optimization

## Requirements:
1. OS Windows/Linux/Mac.
2. Anaconda
3. Python>3.8


### How to use?

#### 1: Setting working environment
Create an environment using the env.yaml in Anaconda with the following command:

conda env create -f env.yaml

#### 2: Data generation
The data generation folder contains files for parametric data generation.

main.py is the main script to run the data generation. 

column.py script contrains the material, geometric, analysis, and model parameters.

functions.py contains function used to generate data, such as section analysis, random generation of section geometry.

The **sample.csv** file contains the sample output from data generation.

#### 3: Data pre-processing and network model
Data preprocesssing script is provided in **h40.py** as an example for height 4.0 meters.

net.py file contrains network file for training the network. 

column.pth file contains saved model for inference. 

#### 4: Design check
Design check folder contains free similar scripts as in data generation.

Use main.py to run the check of the network output results.

