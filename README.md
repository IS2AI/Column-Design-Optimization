# Structural_Column_Design_Optimization

## Requirements:
1. OS Windows/Linux/Mac.
2. Anaconda
3. Python>3.8


### How to use?

#### 1: Setting working environment
Create an environment using the env.yaml in Anaconda with the following command:
conda env create -f environment.yml

#### 2: Data generation
The data generation folder contains files for parametric data generation. 
main.py is the main script to run the data generation. 
column.py script contrains the material, geometric, analysis, and model parameters.
functions.py contains function used to generate data, such as section analysis, random generation of section geometry.

#### 3: Data pre-processing and network model
net.py file contrains network file for training the network. 
column.pth file contains saved model for inference. 

The check folder contains modified files to check the NN-predicted designs for validity following the Eurocode.

The **net.ipynb** contains network model.

Data preprocesssing script is provided in **h40.py** as an example for height 4.0 meters.

The **sample.csv** file contains the sample output from data generation.

#### 4: Design check

