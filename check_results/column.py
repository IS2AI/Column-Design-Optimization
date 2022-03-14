import numpy as np
class Column():
    def __init__(self):
        
        # Cross-section geometry
        self.colHeight = 4.5      # [m]
        self.colWidth_min = 0.2    # [m]
        ### **********Stationary parameters list************************************
   
        #Material constants
        # [m] possible rebars diameters 
        self.d_rebars = [0.12, 0.14, 0.16, 0.02, 0.025, 0.028, 0.032, 0.04, 0.05]
        self.fy = 500 #[MPa]     
        
        # Coefficients
        self.alpha_coef = 0.85
        self.nu_coef = 1.0 #for concrete class less that C50
        self.g_c = 1.5
        self.g_s = 1.15
        
        # Materials for nonlinear columns
        # ------------------------------------------
        # Material ID tags
        
        self.IDcon = 2
        self.IDreinf = 3
        self.lambd = 0.1  # ratio between unloading slope at eps2 and initial slope Ec
        self.Es = 210*1e3 #[MPa]  # Young's modulus
        self.Bs = 0       # strain-hardening ratio
        
        # Unit weight of RC concrete
        self.unit_weight = 0.025   # [MN/m3]
        
        # Column cover to reinforcing steel NA
        self.cover = 0.05      # [m]
        
        self.num_fib = 32 #number of fibers for concrete core and cover in y and z directions
        
        self.SecTag = 1 
        
        # Target ductility for analysis
        self.mu = 15.0
        
        # Number of analysis increments
        self.numIncr = 500