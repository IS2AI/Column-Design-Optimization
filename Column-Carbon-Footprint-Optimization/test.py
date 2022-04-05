import glob
import pandas as pd
import csv
import os
import math
import numpy as np

# Load the dataset
df = pd.read_csv('data.csv')

# Removing the last points where convergence was not achieved
df.sort_values(['My'], ascending=False, inplace = True)
df_drop = df.copy()
df_drop = df_drop.drop_duplicates(subset = ["Width", "Depth", "D_rebar", 'w_g', 'd_g'])

rows = df_drop.index
df.drop(rows, inplace=True)

# Mirror the dataset about the axes
df2 = df.copy()
df2['Mz'] = df['My']
df2['My'] = df['Mz']
df2['Width'] = df['Depth']
df2['Depth'] = df['Width']
df2['w_g'] = df['d_g']
df2['d_g'] = df['w_g']
df.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df = df.append(df2)

# ***********************Selecting short columns******************************
E_list = [31000, 33000, 34000, 35000, 36000, 37000]       # Concrete Young's modulus
df['P'] = (-1)*df['P']
df['fc'] = (-1)*df['fc']

df['fc'] = (-1)*df['fc']
fcs = np.asarray([-50.0, -45.0, -40.0, -35.0, -30.0, -25.0])

# Function for selecting appropriate concrete Young's modulus
def modulus(row):
    if row['E'] == fcs[0]:
        return E_list[0]
    if row['E'] == fcs[1]:
        return E_list[1]
    if row['E'] == fcs[2]:
        return E_list[2]
    if row['E'] == fcs[3]:
        return E_list[3]
    if row['E'] == fcs[4]:
        return E_list[4]
    if row['E'] == fcs[5]:
        return E_list[5]
    return -1

df['E'] = df.apply(lambda row: modulus(row), axis=1)

A = 0.7
B = 1.1
C  = 0.7

df['l_0'] = 0.7*df['h']
df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)

# Compute the slenderness limit
df['n'] = 1.5*df['P']/(df['Ac']*df['fc'])
df['lambda_lim'] = 20*A*B*C/(df['n'])**0.5

# Estimlating slenderness of designs in Y-direction
df['Iy'] = df['Width'] * df['Depth'] ** 3 / 12
df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)
df['lambda_y'] = df['l_0']/(df['Iy']/(df['Width']*df['Depth']))**0.5


# Estimlating slenderness of designs in Z-direction
df['Iz'] = df['Depth'] * df['Width'] ** 3 / 12
df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)
df['lambda_z'] = df['l_0']/(df['Iz']/(df['Width']*df['Depth']))**0.5

# Dropping unnecessary columns
df = df.drop(columns=['E','l_0','Iy','Iz','n','Ac'])

# Dropping slender columns in y direction
df = df[df['lambda_lim']>df['lambda_y']]

# Dropping slender columns in z direction
df = df[df['lambda_lim']>df['lambda_z']]

# Dropping failed cases
df = df[(df['Mz'] > 0.0) & (df['My'] > 0.0)]
df = df[df['P']>0]
df = df.dropna()

# Rounding the axial load to the nearest 100kN, moments to the nearest 100kNm
df['P']=(np.floor(df['P']*10000))/10000
df['My']=(np.floor(df['My']*10000))/10000
df['Mz']=(np.floor(df['Mz']*10000))/10000


# Function for selecting appropriate concrete price by class
def price_concrete(row):
    # Source for prices: https://jcbetons.lv/cenas-en/?lang=en
    # + 21% VAT
    if row['fc'] == fcs[0]:
        return 232
    if row['fc'] == fcs[1]:
        return 254
    if row['fc'] == fcs[2]:
        return 275
    if row['fc'] == fcs[3]:
        return 294
    if row['fc'] == fcs[4]:
        return 311
    if row['fc'] == fcs[5]:
        return 328
    return -1
    

# Define the material prices in EUR/m3
steel_density = 7850 # kg/m3
df['price_s'] = 1.38*steel_density                                    # Steel 
df['price_c'] = df.apply(lambda row: price_concrete(row), axis=1)     # Concrete

# Calculate the cost of designs
df['price'] = df['h']*((df['Width']*df['Depth'] - df['As_total'])*df['price_c'] + df['As_total']*df['price_s'])

# Shuffling the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Copy dataframe for further manipulations
df_copy = df.copy()

# 1. Min-max normalization of P, My, Mz
df[['P', 'My', 'Mz']] = (df[['P', 'My', 'Mz']] - df[['P', 'My', 'Mz']].min()) / (
            df[['P', 'My', 'Mz']].max() - df[['P', 'My', 'Mz']].min())

# 2. Dividing the 3D space (P, My, Mz) into equal sized cubes
# Defining cube size
step_P, step_My, step_Mz = 0.03, 0.03, 0.03

# Adding discretized columns to specify designs in each cube
df['P_dt'] = df['P']-df['P'] % step_P
df['My_dt'] = df['My']-df['My'] % step_My
df['Mz_dt'] = df['Mz']-df['Mz'] % step_Mz

# 3. Backward normalization
df[['P', 'My', 'Mz']] = df[['P', 'My', 'Mz']] * (
            df_copy[['P', 'My', 'Mz']].max()-df_copy[['P', 'My', 'Mz']].min())+df_copy[['P', 'My', 'Mz']].min()

# 4. Price filtration
# Sorting data in each cube by price and maintaining the cheapest designs
df.sort_values(['P_dt', 'My_dt', 'Mz_dt', 'price'], ascending=[True, True, True, True], inplace=True)
df = df.drop_duplicates(subset=['P_dt', 'My_dt', 'Mz_dt'], keep='first')

# Shuffling the dataset after the sort
df = df.sample(frac=1).reset_index(drop=True)

# Dropping unnecessary columns
df = df.drop(columns=['price_s', 'price_c','D_rebar', 'numRebars',
              'My_dt', 'Mz_dt','P_dt', 'w_g','d_g', 'lambda_lim', 'lambda_y', 'lambda_z'])

# Saving dataframe to file
df.to_hdf('pre-processed.h5', 'w')


