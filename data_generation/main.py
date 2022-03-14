import openseespy.opensees as ops
import pandas as pd
import csv
import os
import numpy as np
import random
import math
from functions import *
import column


data = {'P':  [],'My': [],'Mz': [],'Width': [],'Depth': [],'D_rebar': [],
        'w_g': [],'d_g': [],'numRebars': [], 'As_total':[]}
directory = 'D:/output/'

for i in range(10):
    num = i
    results = directory + 'resultsgen_' + str(num)+ '.out'
    logName = directory + 'logs21_' + str(num)+ '.log'
    fileName= directory + 'output/data21_' + str(num)+ '.csv'
    n = 20000 # number of column designs
    numSaveToFile= 1 #number of designs to save 
    
    ### ********************************************************START: Stationary parameters list*************************************************
    parameters = column.Column()
    
    
    ### *************************************START: Outer loop: Iterating over different column designs ***********************************
    ops.logFile(logName,  '-noEcho')  ### To save all logs in the file, instead of terminal
    with open(fileName, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['P','My','Mz', 'Width','Depth','D_rebar','w_g','d_g','numRebars','As_total', 'h', 'fc'])
    
    i=1
    while i < n+1:
        # CONCRETE
        # Nominal concrete compressive strength
        fck = 30                
        
        fc = -fck/parameters.g_c 
        
        eps1U = -0.002      # strain at maximum strength of unconfined concrete from Eurocode
        eps2U = -0.0035     # strain at ultimate stress from Eurocode
    
        # STEEL
        fy = parameters.fy/parameters.g_s      # Yield stress

        # Cross section         
        colWidth, colDepth = cross_section(parameters)
        colHeight = parameters.colHeight
        
     
        print(colHeight, colWidth, colDepth)
        barDiam = random.choice(parameters.d_rebars)
        As_bar = (math.pi)*(barDiam*barDiam)/4
    
        A_core = colWidth*colDepth
        As_min = 0.002*A_core
        As_max = 0.04*A_core
        numRebars_min = math.ceil(As_min/As_bar)
        numRebars_max = math.floor(As_max/As_bar)
        
        # Total number of longitudinal-reinforcement bars in steel layer (symmetric top & bot)
        try:
            
            numBarsSec = random.randint(numRebars_min,numRebars_max)
            if numBarsSec<4:
    
                  continue         
        except:
           
            continue 
       
        
        # Some variables derived from the parameters
        coverY = colDepth/2.0       # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
        coverZ = colWidth/2.0       # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
        coreY = coverY - parameters.cover - barDiam/2     # The distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
        coreZ = coverZ - parameters.cover - barDiam/2     # The distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
        
        dist1 = coverY - parameters.cover/2
        dist2 = coverZ - parameters.cover/2
        ### Scheme with parametrisation using hollow rebar area: w_g, d_g (rebars grid dimensions)
        listDivisorPairs = returnDivisorsPair(numBarsSec)
        if (len(listDivisorPairs) == 1):
            listDivisorPairs = returnDivisorsPair(numBarsSec-1)
            
        # select grid and hollow area parameters
        w_g, d_g= grid_params(listDivisorPairs)
        
        w_h = (colWidth-2*barDiam-2.5*parameters.cover)/colWidth 
        d_h = (colDepth-2*barDiam-2.5*parameters.cover)/colDepth
        
        rebarZ = np.linspace(-coreZ, coreZ, w_g)
        rebarY = np.linspace(-coreY, coreY, d_g)
        spacingZ = (2*coreZ)/(w_g-1)
        spacingY = (2*coreY)/(d_g-1)
    
        # Checking for minimal spacing requirement
        spacing_min=max(2*barDiam, barDiam+0.032+0.005, barDiam+0.020) #[m]
        if (spacingZ < spacing_min or spacingY < spacing_min):
    
            continue
    
        ops.wipe()
        # Define model builder
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # Cover concrete (unconfined)
        ops.uniaxialMaterial('Concrete01', parameters.IDcon, fc, eps1U, fc, eps2U)
        # Reinforcement material     matTag, Fy, E0,  b)
        ops.uniaxialMaterial('Steel01', parameters.IDreinf, fy, parameters.Es, parameters.Bs)

    
        ops.section('Fiber', parameters.SecTag, '-GJ', 1.0e6)
        ops.patch('quadr', parameters.IDcon, parameters.num_fib, parameters.num_fib, -coreY, coreZ, -coreY, -coreZ, coreY, -coreZ, coreY, coreZ)
        # Create the concrete cover fibers (top, bottom, left, right)
        ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coverY, coverZ, -coreY, coreZ, coreY, coreZ, coverY, coverZ)
        ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coreY, -coreZ, -coverY, -coverZ, coverY, -coverZ, coreY, -coreZ)
        ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, -coverY, coverZ, -coverY, -coverZ, -coreY, -coreZ, -coreY, coreZ)
        ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, coreY, coreZ, coreY, -coreZ, coverY, -coverZ, coverY, coverZ)
        
    
        ### Inserting rebars
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
                
    
        ### Check for number of rebars in final configuration
        numTotRebars = len(rebars_YZ)
        if (numTotRebars<4 or numTotRebars*As_bar<As_min) :
            # print("Req-nt on # rebars is not met.")
            continue
        
        # Steel yield strain
        eps = fy/parameters.Es
        
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
        if -0.1*Pmax/fy > As or As<0.002*A_core:
            continue
        
        Pmax = Pmax + parameters.unit_weight*colHeight*colDepth*colWidth
        
        ### *************************************START: Inner loop: Increasing axial load P until failure ***********************************
    
        list_P = np.linspace(0,Pmax,50)
        list_P = np.append(list_P, Pmax)
     
        # First calculate My (uniaxial moment case)
        list_M_maxs=[]
        
        for v in range(len(list_P)):
            # Call the section analysis procedure 
            strain1 = directory + 'strain1_' + str(v)+ '.txt'
            strain2 = directory + 'strain2_' + str(v)+ '.txt'
            strain3 = directory + 'strain3_' + str(v)+ '.txt'
            strain4 = directory + 'strain4_' + str(v)+ '.txt'
            strains = [strain1, strain2, strain3, strain4]
            
            MomentCurvature(parameters, list_P[v], Kz, -1, 5, results, data, strains, dist1, dist2)
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
                M_ult = strain1.loc[Moment_ult, [0]]
                list_M_maxs.append(float(M_ult))
            else:
                M_ult = 0
                list_M_maxs.append(M_ult)
            
            if v>=5:
                myfile1=directory + "strain1_{}.txt".format(v-5)
                myfile2=directory + "strain2_{}.txt".format(v-5)
                myfile3=directory + "strain3_{}.txt".format(v-5)
                myfile4=directory + "strain4_{}.txt".format(v-5)
                
                list_delete = [myfile1, myfile2, myfile3, myfile4]
                ## If file exists, delete it ##
                for myfile in list_delete:
                    if os.path.isfile(myfile):
                        os.remove(myfile)

        # Calculate Mz (biaxial bending case)
        for j in range(len(list_P)):
            P=list_P[j]
            list_m = np.append(list_M_maxs[j]*np.random.random_sample(size = 30 ), list_M_maxs[j])
     
            for m in range(len(list_m)):            
                #Data collection
                data['P'].append(P), data['Width'].append(colWidth), data['Depth'].append(colDepth), data['D_rebar'].append(barDiam)
                data['w_g'].append(w_g), data['d_g'].append(d_g), data['numRebars'].append(numTotRebars),data['As_total'].append(As),
                data['fc'].append(-fck),data['h'].append(colHeight)
                
            
                strain21 = directory + 'strain21_' + str(v)+ '.txt'
                strain22 = directory + 'strain22_' + str(v)+ '.txt'
                strain23 = directory + 'strain23_' + str(v)+ '.txt'
                strain24 = directory + 'strain24_' + str(v)+ '.txt'
                strains2 = [strain21, strain22, strain23, strain24]
                MomentCurvature(parameters, P, Ky, list_m[m], 6, results, data, strains2, dist1, dist2)
                                    
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
                    M_ult = strain1[0].values[Moment_ult]
                    list_M_maxs.append(float(M_ult))
                    data['My'].append(list_m[m])
                    data['Mz'].append(M_ult)
                else:
                    M_ult = 0
                    list_M_maxs.append(M_ult)
                    data['My'].append(list_m[m])
                    data['Mz'].append(M_ult)
                
                if v>=5:
                    myfile1=directory + "strain21_{}.txt".format(v-5)
                    myfile2=directory + "strain22_{}.txt".format(v-5)
                    myfile3=directory + "strain23_{}.txt".format(v-5)
                    myfile4=directory + "strain24_{}.txt".format(v-5)
                    
                    list_delete = [myfile1, myfile2, myfile3, myfile4]
                    for myfile in list_delete:
                        if os.path.isfile(myfile):
                            os.remove(myfile)
    
        if i%numSaveToFile == 0:
            df = pd.DataFrame(data)
            df=df[(df['Mz'].astype(float)>0.0) & (df['My'].astype(float) > 0.0)]
            df = df.dropna()
            
            df.to_csv(fileName, mode='a', index=False, header=False)
            print("#%s: %s column designs already saved."%(num, i) )
            data = {'P':[],'My':[],'Mz':[],'Width':[],'Depth':[],'D_rebar':[],'w_g':[],'d_g':[],'numRebars':[],'As_total':[], 'h':[],'fc':[]}
        i+=1

    
df.to_csv(fileName, mode='a', index=False, header=False)





