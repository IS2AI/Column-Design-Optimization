# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:44:33 2022

@author: aknur
"""
import openseespy.opensees as ops
import pandas as pd
import csv
import os
import numpy as np
import random
import math
from functions import *
import column
import time
start_time = time.time()

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Create an instance of class Column 
parameters = column.Column()

# Loading the dataset
df = pd.read_csv("nn_combo1.csv")
#df = df[1:]
# Adding columns to store gridlines
df['w_g']=0
df['d_g']=0

# Directory to store the output response files
directory ='.'

 
eps1U = -0.002                         # strain at maximum strength of unconfined concrete from Eurocode
eps2U = -0.0035                        # strain at ultimate stress from Eurocode
fy = parameters.fy/parameters.g_s      # Yield stress

# Lists to track the checks
Mz_check = []
My_check = []
As_final = []
As_1 = []
As_2 = []
P_max = []
Spacing = []
numBar = []


# Iterating over the test case prediction designs
for index, row in df.iterrows():
    print(index)
    
    # Reading the predictions, put the dataset columns in the following order
    colHeight,fc, P, My_red, Mz_red,colWidth, colDepth,As_total, w_g, d_g = row

    fc = fc/parameters.g_c           # Concrete design strength
    
    # Rounding down the predicted geometry dimensions
    colWidth = 0.05*math.floor(colWidth/0.05)
    colDepth = 0.05*math.floor(colDepth/0.05)


    P = -P
    count = 0
    passed_check = 0
    while passed_check==0:      
        #passed_check=1
        # If rounding down make the design fail, round up once 
        if count==1:
            #colWidth = colWidth + 0.05
            colDepth = colDepth + 0.05
            print(colWidth, colDepth)
            count = 2
        count = 1
             
        # Iterate over all rebar diameters starting from largest one
        # since it is more practical from engineering perspective
        for barDiam in parameters.d_rebars:
            barDiam = int(1000*barDiam)/1000
            As_bar = (math.pi)*(barDiam*barDiam)/4
            A_core = colWidth*colDepth
            As_min = 0.002*A_core
            As_max = 0.04*A_core
            
            # Check the steel area requirements
            if As_total>=As_min and As_total<As_max:
                As_1.append(index)
                numBarsSec = math.ceil(As_total/As_bar)
                
                # Check for the number of reinforcements
                if  numBarsSec>= 4:
                    numBar.append(index)
                
                    # Some variables derived from the parameters
                    coverY = colDepth/2.0       # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
                    coverZ = colWidth/2.0       # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
                    coreY = coverY - parameters.cover - barDiam/2     # The distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
                    coreZ = coverZ - parameters.cover - barDiam/2     # The distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
                    
                    dist1 = coverY - parameters.cover/2
                    dist2 = coverZ - parameters.cover/2
                    
                    # Scheme with parametrisation using no reinforcement area
                    listDivisorPairs = returnDivisorsPair(numBarsSec)
                    if (len(listDivisorPairs) == 1):
                        listDivisorPairs = returnDivisorsPair(numBarsSec-1)
                          
                    list_w_g = [*range(2, math.ceil(numBarsSec/2)+1, 1)]
                    
                    for w_g in list_w_g:
                        d_g = math.ceil(numBarsSec/2)-w_g+2
                
                        w_h = (colWidth-2*barDiam-2.5*parameters.cover)/colWidth 
                        d_h = (colDepth-2*barDiam-2.5*parameters.cover)/colDepth
                        
                        rebarZ = np.linspace(-coreZ, coreZ, w_g)
                        rebarY = np.linspace(-coreY, coreY, d_g)
                        spacingZ = (2*coreZ)/(w_g-1)
                        spacingY = (2*coreY)/(d_g-1)
                        
                        # Checking for minimal spacing requirement
                        spacing_min=max(2*barDiam, barDiam+0.032+0.005, barDiam+0.020) #[m]
                        if (spacingZ > spacing_min or spacingY > spacing_min):
                            Spacing.append(index)
                    
                            ops.wipe()
                            # Define model builder
                            ops.model('basic', '-ndm', 3, '-ndf', 6)
                            
                            # Cover concrete (unconfined)
                            ops.uniaxialMaterial('Concrete01', parameters.IDcon, fc, eps1U, fc, eps2U)
                            # Reinforcement material     matTag, Fy, E0,  b)
                            ops.uniaxialMaterial('Steel01', parameters.IDreinf, fy, parameters.Es, parameters.Bs)
                        
                            # Construct concrete fibers
                            ops.section('Fiber', parameters.SecTag, '-GJ', 1.0e6)
                            ops.patch('quadr', parameters.IDcon, parameters.num_fib, parameters.num_fib, -coreY, coreZ, -coreY, -coreZ, coreY, -coreZ, coreY, coreZ)
                            ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coverY, coverZ, -coreY, coreZ, coreY, coreZ, coverY, coverZ)
                            ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coreY, -coreZ, -coverY, -coverZ, coverY, -coverZ, coreY, -coreZ)
                            ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, -coverY, coverZ, -coverY, -coverZ, -coreY, -coreZ, -coreY, coreZ)
                            ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, coreY, coreZ, coreY, -coreZ, coverY, -coverZ, coverY, coverZ)
                            
                            # Inserting rebars
                            hollowY = d_h*coverY
                            hollowZ = w_h*coverZ
                        
                            rebars_YZ = np.empty((0,2))
                        
                            for ii, Y in enumerate(rebarY):
                                for jj, Z in enumerate(rebarZ):
                                    if (abs(Y) < hollowY and abs(Z) < hollowZ):
                                        continue
                                    rebars_YZ = np.vstack([rebars_YZ, [Y,Z]])
                            for ii in range(len(rebars_YZ)):
                                    ops.fiber(*rebars_YZ[ii], As_bar, parameters.IDreinf)
                                    
                        
                            # Check for number of rebars in final configuration
                            numTotRebars = len(rebars_YZ)
                            if (numTotRebars>4 or numTotRebars*As_bar>As_min) :
                                As_2.append(index)
                                
                                eps = fy/parameters.Es       # Steel yield strain
                                
                                # Yield curvature
                                # d -- from cover to rebar 
                                d_z = colDepth-parameters.cover-barDiam/2
                                Kz = eps/(0.7*d_z)
                                
                                d_y = colWidth-parameters.cover-barDiam/2
                                Ky = eps/(0.7*d_y)
                            
                                # Set axial load
                                As = As_bar * numTotRebars
                                Ac = colWidth*colDepth - As
                                Pmax = -parameters.alpha_coef*(parameters.nu_coef*(-fc)*Ac + fy*As)
                                Pmax = Pmax + parameters.unit_weight*colHeight*colDepth*colWidth
                                
                                # Record the bar diameter
                                df.at[index, 'D_rebar'] = barDiam
                                numRebars = w_g*2+(d_g-2)*2
                                # Record number of rebars
                                df.at[index, 'numRebars'] = numRebars
                                
                                # Check for finas steel area
                                if -0.1*Pmax/fy < As or As>0.002*A_core:
                                    # Record final steel area check pass
                                    As_final.append(index)
                                    
                                    # Check for axial load capacity
                                    if Pmax<P:
                                        # Record the axial load check pass
                                        P_max.append(index)
                                        
                                        strain1 = directory + 'strain1_.txt'
                                        strain2 = directory + 'strain2_.txt'
                                        strain3 = directory + 'strain3_.txt'
                                        strain4 = directory + 'strain4_.txt'
                                        strains = [strain1, strain2, strain3, strain4]
                                        
                                        # Call the section analysis procedure 
                                        MomentCurvature(parameters, P, Kz, -1, 5, strains, dist1, dist2)
                                        indeces = []
                                        
                                        if os.path.getsize(strain1)>0:
                                            strain1 = pd.read_csv(strain1, sep = ' ', header = None, )
                                            filtered1 = strain1[strain1[2]>= -0.0035]
                                            if len(filtered1)> 1:
                                                indeces.append(list(filtered1.index)[-1])
                                            
                                        if os.path.getsize(strain2)>0:
                                            strain2 = pd.read_csv(strain2, sep = ' ', header = None, )
                                            filtered2 = strain2[strain2[2]>= -0.0035]
                                            if len(filtered2)> 1:
                                                indeces.append(list(filtered2.index)[-1])
                                        
                                        if os.path.getsize(strain3)>0:
                                            strain3 = pd.read_csv(strain3, sep = ' ', header = None, )
                                            filtered3 = strain3[strain3[2]>= -0.0035]
                                            if len(filtered3)> 1:
                                                indeces.append(list(filtered3.index)[-1])
                                        
                                        if os.path.getsize(strain4)>0:
                                            strain4 = pd.read_csv(strain4, sep = ' ', header = None, )
                                            filtered4 = strain4[strain4[2]>= -0.0035]
                                            if len(filtered4)> 1:
                                                indeces.append(list(filtered4.index)[-1])
                                            
                                        if len(indeces)>=1:
                                            Moment_ult = min(indeces)
                                            My_max = strain1.loc[Moment_ult, [0]]
                                            My_max = My_max[0]
                                            
                                            if My_max> My_red:
                                                # Record Mz pass check
                                                My_check.append(index)
                                                
                                                strain21 = directory + 'strain21_.txt'
                                                strain22 = directory + 'strain22_.txt'
                                                strain23 = directory + 'strain23_.txt'
                                                strain24 = directory + 'strain24_.txt'
                                                strains2 = [strain21, strain22, strain23, strain24]
                                                
                                                # Call the section analysis procedure
                                                MomentCurvature(parameters, P, Ky, My_red, 6, strains2, dist1, dist2)
                                                                    
                                                indeces = []
                                                if os.path.getsize(strain21)>0:
                                                    strain1 = pd.read_csv(strain21, sep = ' ', header = None)
                                                    filtered1 = strain1[strain1[2]>= -0.0035]
                                                    if len(filtered1)> 1:
                                                        indeces.append(list(filtered1.index)[-1])
                                                    
                                                if os.path.getsize(strain22)>0:
                                                    strain2 = pd.read_csv(strain22, sep = ' ', header = None)
                                                    filtered2 = strain2[strain2[2]>= -0.0035]
                                                    if len(filtered2)> 1:
                                                        indeces.append(list(filtered2.index)[-1])
                                                
                                                if os.path.getsize(strain23)>0:
                                                    strain3 = pd.read_csv(strain23, sep = ' ', header = None)
                                                    filtered3 = strain3[strain3[2]>= -0.0035]
                                                    if len(filtered3)> 1:
                                                        indeces.append(list(filtered3.index)[-1])
                                                
                                                if os.path.getsize(strain24)>0:
                                                    strain4 = pd.read_csv(strain24, sep = ' ', header = None)
                                                    filtered4 = strain4[strain4[2]>= -0.0035]
                                                    if len(filtered4)> 1:
                                                        indeces.append(list(filtered4.index)[-1])
                                                    
                                                if len(indeces)>=1:
                                                    Moment_ult = min(indeces)
                                                    Mz_max = strain1.loc[Moment_ult, [0]]
                                                    Mz_max = Mz_max[0]
                                                    
                                                    # Check for Mz capacity
                                                    if Mz_max> Mz_red:
                                                        Mz_check.append(index)
                                                        
                                                        #Record final design configuration
                                                        df.at[index, 'Width'] = colWidth
                                                        df.at[index, 'Depth'] = colDepth
                                                        df.at[index, 'w_g'] = w_g
                                                        df.at[index, 'd_g'] = d_g 
                                                        numRebars = w_g*2+(d_g-2)*2                                                       
                                                        df.at[index, 'numRebars'] = numRebars
                                                        df.at[index, 'As_total'] = As
                                                        
                                                        print("#################################")
                                                        passed_check = 1
                                                        break
                    else:
                        
                        continue
                    break
            
          
print( len(set(As_1)), len(set(As_2)), len(set(As_final)), len(set(numBar)), len(set(P_max)), len(set(My_check)), len(set(Mz_check)))               
       
# Compute the cost of the final designs
#df['fc'] = (-1)*df['fc']
fcs = np.asarray([-50.0, -45.0, -40.0, -35.0, -30.0, -25.0])

def price_concrete(row):
    # Source for prices: https://jcbetons.lv/cenas-en/?lang=en
    # + 21% VAT
    if row['fc'] == fcs[0]:
        return 232  # 95 EUR/m3 - assumed value
    if row['fc'] == fcs[1]:
        return 254  # 90 EUR/m3 - assumed value
    if row['fc'] == fcs[2]:
        return 275  # 85 EUR/m3 - assumed value
    if row['fc'] == fcs[3]:
        return 294
    if row['fc'] == fcs[4]:
        return 311
    if row['fc'] == fcs[5]:
        return 328
    return -1

# Prices in EUR/m3
df['price_s'] = 1.38*7850
df['price_c'] = df.apply(lambda row: price_concrete(row), axis=1)

df['price'] = df['colHeight']*((df['Width']*df['Depth'] - df['As_total'])*df['price_c'] + df['As_total']*df['price_s'])
            
print("--- %s seconds ---" % (time.time() - start_time))
        
                
          
     