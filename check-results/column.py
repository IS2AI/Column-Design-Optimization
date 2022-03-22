import numpy as np
class Column():
    def __init__(self):
        # The class Column contains all the static material, geometry and analysis parameters.
        
        # Geometric parameters
        self.colHeight = 3.0                     # Column height [m]
        self.colWidth_min = 0.2                  # Column minimum width [m]
        
        # List of rebar diameters [m] 
        self.d_rebars = np.asarray([0.012, 0.014, 0.016, 0.020, 0.025, 0.028, 0.032, 0.040, 0.050]) 
        
        # Eurocode design parameters
        self.alpha_coef = 0.85                   # Longterm effect of compressive strength
        self.nu_coef = 1.0                       # Effective strength factor [fc<= C50]
        self.g_c = 1.5                           # Concrete safety factor
        self.g_s = 1.15                          # Steel safety factor
        
        # Steel material parameters
        self.IDreinf = 3                         # Steel reinforcement material ID tag
        self.Es = 210*1e3                        # Young's modulus of steel  [MPa]
        self.Bs = 0                              # Steel strain-hardening ratio
        self.fy = 500                            # Strength of steel [MPa]     
        
        # Concrete material parameters
        self.IDcon = 2                           # Concrete material ID tag
        self.unit_weight = 0.025                 # Unit weight of RC concrete [MN/m3]
        self.eps1U = -0.002                      # Strain at maximum strength of unconfined concrete 
        self.eps2U = -0.0035                     # Strain at ultimate stress 
    
        # Geometry and analysis parameters
        self.cover = 0.05                        # Column cover [m]
        self.num_fib = 16                        # Number of fibers in y and z directions
        self.SecTag = 1                          # Column section
        self.mu = 15.0                           # Target ductility for analysis
        self.numIncr = 500                       # Number of analysis increments