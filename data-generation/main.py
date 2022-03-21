import openseespy.opensees as ops
import pandas as pd
import csv
import os
import numpy as np
import random
import math
from functions import *
import column


# Create a dictionary to store the column section design parameters
data = {'P':  [],'My': [],'Mz': [],'Width': [],'Depth': [],'D_rebar': [],
        'w_g': [],'d_g': [],'numRebars': [], 'As_total':[], 'h':[], 'fc':[]}

# Directory to store the ouput files
directory = 'D:/output/'
n = 1                                           # Number of column designs
numSaveToFile= 1                                # Number of designs to save at a time 

# Creating a file to store the opensees logs and cash 
logName = directory + 'logs.log' 

# Crearing a csv file to store the generated dataset
fileName= directory + 'output/data.csv'

# Create an object of class Columns to call the material, geometry and analysis parameters
parameters = column.Column()              

ops.logFile(logName,  '-noEcho')               # Send all logs to the logName file instead of terminal

# Start writing the design data to the csv file
with open(fileName, 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['P','My','Mz', 'Width','Depth','D_rebar','w_g','d_g','numRebars','As_total', 'h', 'fc'])


i=1

# *************START: Outer loop for parametric designs*********************
while i < n+1:
    
    # Concrete parameters   
    fck = 30                                  # Characteristic strength of concrete
    fc = -fck/parameters.g_c                  # Design strength of concrete      
    eps1U = parameters.eps1U                  # Strain at maximum strength
    eps2U = parameters.eps1U                  # Strain at ultimate strength

    # Steel parameters
    fy = parameters.fy/parameters.g_s         # Design strength of steel 

    # Randomly generate column cross-section         
    colWidth, colDepth = cross_section(parameters)
    colHeight = parameters.colHeight          # Column height

    print(colHeight, colWidth, colDepth)
    
    # Select reinforcement diameter
    barDiam = random.choice(parameters.d_rebars)
    
    # Calculate the area of one rebar
    As_bar = (math.pi)*(barDiam*barDiam)/4

    # Calculate the steel area and reinforcement constraints
    A_core = colWidth*colDepth                # Section gross area
    As_min = 0.002*A_core                     # Minimum area of steel
    As_max = 0.04*A_core                      # Maximum area of steel
    numRebars_min = math.ceil(As_min/As_bar)  # Minimum number of rebars
    numRebars_max = math.floor(As_max/As_bar) # Maximum number of rebars
    
    # Total number of longitudinal-reinforcement bars 
    try:
        
        numBarsSec = random.randint(numRebars_min,numRebars_max)
        if numBarsSec<4:

              continue         
    except:
       
        continue 
   
    
    # Section geometry modelling parameters 
    coverY = colDepth/2.0       # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
    coverZ = colWidth/2.0       # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
    coreY = coverY - parameters.cover - barDiam/2     # The distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
    coreZ = coverZ - parameters.cover - barDiam/2     # The distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
    
    dist1 = coverY - parameters.cover/2
    dist2 = coverZ - parameters.cover/2
    
    # Generating the grid parameters for rebars placement
    listDivisorPairs = returnDivisorsPair(numBarsSec)
    if (len(listDivisorPairs) == 1):
        listDivisorPairs = returnDivisorsPair(numBarsSec-1)

    w_g, d_g= grid_params(listDivisorPairs)   
    
    # No reinforcement area, to place reinforcement along the perimeter of section
    w_h = (colWidth-2*barDiam-2.5*parameters.cover)/colWidth  
    d_h = (colDepth-2*barDiam-2.5*parameters.cover)/colDepth
    
    rebarZ = np.linspace(-coreZ, coreZ, w_g)         # Coordinates of rebars in Z axis
    rebarY = np.linspace(-coreY, coreY, d_g)         # Coordinates of rebars in Y axis
    spacingZ = (2*coreZ)/(w_g-1)                     # Spacing between rebars in Z axis
    spacingY = (2*coreY)/(d_g-1)                     # Spacing between rebars in Y axis

    # Checking for reinforcement bars minimum spacing requirement
    spacing_min=max(2*barDiam, barDiam+0.032+0.005, barDiam+0.020) # Minimum allowable spacing [m]
    if (spacingZ < spacing_min or spacingY < spacing_min):

        continue

    # Clean the cash and saved parameters from previous design
    ops.wipe()
    
    # Define model builder
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Create concrete material
    ops.uniaxialMaterial('Concrete01', parameters.IDcon, fc, eps1U, fc, eps2U)
    # Create steel material
    ops.uniaxialMaterial('Steel01', parameters.IDreinf, fy, parameters.Es, parameters.Bs)

    # Create section 
    ops.section('Fiber', parameters.SecTag, '-GJ', 1.0e6)
    # Construct fibers in the concrete core
    ops.patch('quadr', parameters.IDcon, parameters.num_fib, parameters.num_fib, -coreY, coreZ, -coreY, -coreZ, coreY, -coreZ, coreY, coreZ)
    # Construct fibers in the concrete cover at four sides
    ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coverY, coverZ, -coreY, coreZ, coreY, coreZ, coverY, coverZ)
    ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coreY, -coreZ, -coverY, -coverZ, coverY, -coverZ, coreY, -coreZ)
    ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, -coverY, coverZ, -coverY, -coverZ, -coreY, -coreZ, -coreY, coreZ)
    ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, coreY, coreZ, coreY, -coreZ, coverY, -coverZ, coverY, coverZ)
    

    # Inserting rebars along the perimeter of the section
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
            

    # Check for number of rebars in final configuration, should not be less than 4
    numTotRebars = len(rebars_YZ)
    if (numTotRebars<4 or numTotRebars*As_bar<As_min) :
        # print("Req-nt on # rebars is not met.")
        continue
    
    # Steel yield strain
    eps = fy/parameters.Es
    
    d_z = colDepth-parameters.cover-barDiam/2     # Distance from column outer edge to rebar 
    Kz = eps/(0.7*d_z)                            # Yield curvature in Z direction
    
    d_y = colWidth-parameters.cover-barDiam/2     # Distance from column outer edge to rebar
    Ky = eps/(0.7*d_y)                            # Yield curvature in Y direction

    # Compute the axial load capacity
    As = As_bar * numTotRebars
    Ac = colWidth*colDepth - As
    Pmax = -parameters.alpha_coef*(parameters.nu_coef*(-fc)*Ac + fy*As)
    
    # Check for steel area requirement
    if -0.1*Pmax/fy > As or As<0.002*A_core:
        continue
    
    # Computer the axial load capacity for the bottom section of column
    Pmax = Pmax + parameters.unit_weight*colHeight*colDepth*colWidth

    # **********START: Inner loop: Increasing axial load up to Pmax ***********

    # Generate list of axial load P
    list_P = np.linspace(0,Pmax,50)
    
 
    # First call analysis to calculate My capacity (uniaxial moment case)
    
    # List to store the My capacity for each axial load P from list_P
    list_M_maxs=[]
    
    for v in range(len(list_P)):
        # Create files to store the stress, strain and moment from 
        # four corner points of column section
        strain1 = directory + 'strain1_' + str(v)+ '.txt'
        strain2 = directory + 'strain2_' + str(v)+ '.txt'
        strain3 = directory + 'strain3_' + str(v)+ '.txt'
        strain4 = directory + 'strain4_' + str(v)+ '.txt'
        strains = [strain1, strain2, strain3, strain4]
        
        # Call the section analysis procedure
        MomentCurvature(parameters, list_P[v], Kz, -1, 5, strains, dist1, dist2)
        
        # Create a list to store the step when the ultimate strength strain is reached
        indices = []
        
        # Extract the step when the first corner point reached the ultimate strain
        if os.path.getsize(strain1)>0:
            strain1 = pd.read_csv(strain1, sep = ' ', header = None, )
            filtered1 = strain1[strain1[2]>=-0.0035]
            if len(filtered1)> 1:
                indices.append(list(filtered1.index)[-1])
          
        # Extract the step when the second corner point reached the ultimate strain
        if os.path.getsize(strain2)>0:
            strain2 = pd.read_csv(strain2, sep = ' ', header = None, )
            filtered2 = strain2[strain2[2]>=-0.0035]
            if len(filtered2)> 1:
                indices.append(list(filtered2.index)[-1])
        
        # Extract the step when the third corner point reached the ultimate strain
        if os.path.getsize(strain3)>0:
            strain3 = pd.read_csv(strain3, sep = ' ', header = None, )
            filtered3 = strain3[strain3[2]>=-0.0035]
            if len(filtered3)> 1:
                indices.append(list(filtered3.index)[-1])
        
        # Extract the step when the forth corner point reached the ultimate strain
        if os.path.getsize(strain4)>0:
            strain4 = pd.read_csv(strain4, sep = ' ', header = None, )
            filtered4 = strain4[strain4[2]>=-0.0035]
            if len(filtered4)> 1:
                indices.append(list(filtered4.index)[-1])
            
        # Extract the step when one of the four edge points reached the ultimate
        # strain first
        if len(indices)>=1:
            Moment_ult = min(indices)
            M_ult = strain1.loc[Moment_ult, [0]]
            list_M_maxs.append(float(M_ult))
        else:
            # if convergence wasn't reached set moment capacity to zero
            M_ult = 0
            list_M_maxs.append(M_ult)
        
        # Delete the files with the stress, strain and moment to free the memory
        if v>=5:
            myfile1=directory + "strain1_{}.txt".format(v-5)
            myfile2=directory + "strain2_{}.txt".format(v-5)
            myfile3=directory + "strain3_{}.txt".format(v-5)
            myfile4=directory + "strain4_{}.txt".format(v-5)
            
            list_delete = [myfile1, myfile2, myfile3, myfile4]
            for myfile in list_delete:
                if os.path.isfile(myfile):
                    os.remove(myfile)

    # Call analysis to calculate Mz capacity (biaxial moment case)
    # Iterate for each axial load P
    for j in range(len(list_P)):
        P=list_P[j]
        
        # Create a list of moments in Y direction up to My capacity
        list_m = np.append(list_M_maxs[j]*np.random.random_sample(size=29), list_M_maxs[j])
        
        # Iterate for each axial load P and moment My
        for m in range(len(list_m)):            
            # Fill the dictionary with the current design parameters
            data['P'].append(P), data['Width'].append(colWidth), data['Depth'].append(colDepth), data['D_rebar'].append(barDiam)
            data['w_g'].append(w_g), data['d_g'].append(d_g), data['numRebars'].append(numTotRebars),data['As_total'].append(As),
            data['h'].append(colHeight),data['fc'].append(-fck)
            
            # Create files to store the stress, strain and moment from 
            # four corner points of column section for biaxial bending case
            strain21 = directory + 'strain21_' + str(v)+ '.txt'
            strain22 = directory + 'strain22_' + str(v)+ '.txt'
            strain23 = directory + 'strain23_' + str(v)+ '.txt'
            strain24 = directory + 'strain24_' + str(v)+ '.txt'
            strains2 = [strain21, strain22, strain23, strain24]
            
            # Call the section analysis procedure to computer Mz capacity
            MomentCurvature(parameters, P, Ky, list_m[m], 6, strains2, dist1, dist2)
                                
            # Reset a list to store the step when the ultimate strength strain is reached
            indices = []
        
            # Extract the step when the first corner point reached the ultimate strain
            if os.path.getsize(strain21)>0:
                strain1 = pd.read_csv(strain21, sep = ' ', header = None)
                filtered1 = strain1[strain1[2]>= -0.0035]
                if len(filtered1)> 1:
                    indices.append(list(filtered1.index)[-1])
                
            # Extract the step when the second corner point reached the ultimate strain
            if os.path.getsize(strain22)>0:
                strain2 = pd.read_csv(strain22, sep = ' ', header = None)
                filtered2 = strain2[strain2[2]>= -0.0035]
                if len(filtered2)> 1:
                    indices.append(list(filtered2.index)[-1])
            
            # Extract the step when the third corner point reached the ultimate strain
            if os.path.getsize(strain23)>0:
                strain3 = pd.read_csv(strain23, sep = ' ', header = None)
                filtered3 = strain3[strain3[2]>= -0.0035]
                if len(filtered3)> 1:
                    indices.append(list(filtered3.index)[-1])
            
            # Extract the step when the forth corner point reached the ultimate strain
            if os.path.getsize(strain24)>0:
                strain4 = pd.read_csv(strain24, sep = ' ', header = None)
                filtered4 = strain4[strain4[2]>= -0.0035]
                if len(filtered4)> 1:
                    indices.append(list(filtered4.index)[-1])
            
            # Extract the step when one of the four edge points reached the ultimate
            # strain first
            if len(indices)>=1:
                Moment_ult = min(indices)
                M_ult = strain1[0].values[Moment_ult]
                list_M_maxs.append(float(M_ult))
                data['My'].append(list_m[m])
                data['Mz'].append(M_ult)
            else:
                M_ult = 0
                list_M_maxs.append(M_ult)
                data['My'].append(list_m[m])
                data['Mz'].append(M_ult)
            
            # Delete the files with the stress, strain and moment to free the memory
            if v>=5:
                myfile1=directory + "strain21_{}.txt".format(v-5)
                myfile2=directory + "strain22_{}.txt".format(v-5)
                myfile3=directory + "strain23_{}.txt".format(v-5)
                myfile4=directory + "strain24_{}.txt".format(v-5)
                
                list_delete = [myfile1, myfile2, myfile3, myfile4]
                for myfile in list_delete:
                    if os.path.isfile(myfile):
                        os.remove(myfile)

    # Save the design
    if i%numSaveToFile == 0:
        # Create dataframe with the data from dictionary of design parameters
        df = pd.DataFrame(data)
        
        # Drop failure points 
        df=df[(df['Mz'].astype(float)>0.0) & (df['My'].astype(float) > 0.0)]
        df = df.dropna()  
        
        # Save the dataframe with designs to a csv file
        df.to_csv(fileName, mode='a', index=False, header=False)
        print("%s column designs already saved."%(i) )
        
        # Clean the disctionary 
        data = {'P':[],'My':[],'Mz':[],'Width':[],'Depth':[],'D_rebar':[],'w_g':[],'d_g':[],'numRebars':[],'As_total':[], 'h':[],'fc':[]}
    
    # Increase counter by one for the next design
    i+=1



