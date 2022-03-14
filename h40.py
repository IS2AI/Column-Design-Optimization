import glob
import pandas as pd
import csv
import os
import math
import numpy as np

height = ['h40']
fcs = ['fc25', 'fc30','fc35', 'fc40', 'fc45', 'fc50']

for h in height:
    for i in range(len(fcs)):
        fc = fcs[i]
        directory = h+'/'+fc+'/'
        print(directory)
        df = pd.concat(map(pd.read_csv, glob.glob(directory+'*.csv')))
        #df.to_hdf(directory+"all.h5", 'w')

        df.sort_values(['My'], ascending=False, inplace = True)

        df_drop = df.copy()
        df_drop = df_drop.drop_duplicates(subset = ["Width", "Depth", "D_rebar", 'w_g', 'd_g'])


        rows = df_drop.index

        print(len(df))
        df.drop(rows, inplace=True)
        print(len(df))

        df.sort_values(['My'], ascending=False, inplace = True)
        df_drop2 = df.copy()
        df_drop2 = df_drop2.drop_duplicates(subset = ["Width", "Depth", "D_rebar", 'w_g', 'd_g'])

        rows = df_drop2.index

        print(len(df))
        df.drop(rows, inplace=True)
        print(len(df))


        d_w = df[df['Width']<df['Depth']]

        w_d = df[df['Width']>df['Depth']]
        print(len(d_w), len(w_d))


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
        len(df)

        df.to_hdf(directory+'merged.h5', 'w')
        
##### Selecting short columns
        E_list = [31000, 33000, 34000, 35000, 36000, 37000]
        df['P'] = (-1)*df['P']
        df['fc'] = (-1)*df['fc']
        df['E'] = E_list[i]

        # lambda_lim = 20ABC/sqrt(n)

        A = 0.7
        B = 1.1
        C  = 0.7

        df['l_0'] = 0.7*df['h']
        df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)
        # lambda = l_0/srqt(I/A)

        df['n'] = 1.5*df['P']/(df['Ac']*df['fc'])
        df['lambda_lim'] = 20*A*B*C/(df['n'])**0.5

        # Y-direction
        df['Iy'] = df['Width'] * df['Depth'] ** 3 / 12
        df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)
        df['lambda_y'] = df['l_0']/(df['Iy']/(df['Width']*df['Depth']))**0.5


        # Z-direction
        df['Iz'] = df['Depth'] * df['Width'] ** 3 / 12
        df['Ac'] = df['Width']*df['Depth']-(3.14*df['numRebars']*(df['D_rebar']**2)/4)
        df['lambda_z'] = df['l_0']/(df['Iz']/(df['Width']*df['Depth']))**0.5

        df = df.drop(columns=['E','l_0','Iy','Iz','n','Ac'])

        print(len(df))
        df = df[df['lambda_lim']>df['lambda_y']]

        df = df[df['lambda_lim']>df['lambda_z']]
        print(len(df))

### Price estimation
        # Dropping failed cases
        df = df[(df['Mz'] > 0.0) & (df['My'] > 0.0)]
        df = df[df['P']>0]
        df = df.dropna()

        df['P']=(np.floor(df['P']*10000))/10000
        df['My']=(np.floor(df['My']*10000))/10000
        df['Mz']=(np.floor(df['Mz']*10000))/10000


        price_conc_list = [60, 63, 72, 85, 90, 95]
        price_conc_list = [x*1.21 for x in price_conc_list]
        
        # Prices in EUR/m3
        df['price_s'] = 0.31* 1.21*7850
        df['price_c'] = price_conc_list[i]

        df['price'] = df['h']*((df['Width']*df['Depth'] - df['As_total'])*df['price_c'] + df['As_total']*df['price_s'])
        
        df.to_hdf(directory+"with_price.h5", key='w')
        df = df.sample(frac=1).reset_index(drop=True)

        df_copy = df.copy()
    
        # 1. Min-max normalization P, My, Mz: P = (P - Pmin)/(Pmax - Pmin)
        df[['P', 'My', 'Mz']] = (df[['P', 'My', 'Mz']] - df[['P', 'My', 'Mz']].min()) / (
                    df[['P', 'My', 'Mz']].max() - df[['P', 'My', 'Mz']].min())

        # 2. Dividing the 3D space (P, My, Mz) into equal sized cubes
        # Discretization steps
        step_P, step_My, step_Mz = 0.03, 0.03, 0.03
        # Adding discretized columns
        df['P_dt'] = df['P']-df['P'] % step_P
        df['My_dt'] = df['My']-df['My'] % step_My
        df['Mz_dt'] = df['Mz']-df['Mz'] % step_Mz

        # 3. Backward normalization
        df[['P', 'My', 'Mz']] = df[['P', 'My', 'Mz']] * (
                    df_copy[['P', 'My', 'Mz']].max()-df_copy[['P', 'My', 'Mz']].min())+df_copy[['P', 'My', 'Mz']].min()

        # 4. Price filtration
        # Sorting data in each cube by price
        df.sort_values(['P_dt', 'My_dt', 'Mz_dt', 'price'], ascending=[True, True, True, True], inplace=True)
        df = df.drop_duplicates(subset=['P_dt', 'My_dt', 'Mz_dt'], keep='first')

        df = df.sample(frac=1).reset_index(drop=True)
        
        df = df.drop(columns=['price_s', 'price_c','D_rebar', 'numRebars',
                      'My_dt', 'Mz_dt','P_dt', 'w_g','d_g', 'lambda_lim', 'lambda_y', 'lambda_z'])
        df.to_hdf(directory+'price.h5', 'w')




