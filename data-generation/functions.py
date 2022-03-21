import openseespy.opensees as ops
import math
import random

def cross_section(parameters):
    '''
    This function randomly generate section width and depth. According to 5.3.1.7 Eurocode 2, 
    A column is a member for which the section depth does not exceed 4 times its width and
    the height is at least 3 times the section depth. Otherwise it should be considered as a wall.
    The input is object of class parameters that contains the geomteric configurations.
    Outputs are Width, Depth.
    '''
    colWidthArray = [*range(round(0.2/0.05), round((3/3+0.05)/0.05), 1)]
    colWidthArray  = [round(0.05*x,2) for x in colWidthArray]
    colWidth = random.choice(colWidthArray)
   
    colDepthArray = [*range(round(max(0.2,colWidth/4)/0.05), round((min(colWidth*4, 3/3)+0.05)/0.05),1)]
    colDepthArray = [round(0.05*x,2) for x in colDepthArray] 
    colDepth = random.choice(colDepthArray)
    
    Depth = max(colWidth, colDepth)
    Width = min(colWidth, colDepth)
    return Width, Depth

# Degree of fredom (dof) = 5 for uniaxial moment
# Degree of fredom (dof) = 6 for biaxial moment
def MomentCurvature(parameters, axialLoad, maxK, m, dof, strains2, dist1, dist2):
    '''
    This is the function for column section analysis. For uniaxial case degree
    of freedom dof=5, for biaxial case dof=6. Inputs are object of class parameters, 
    axial load P, maximum curvature, first direction moment, degree of freedom,
    list of txt files, the absolute value of edge point 
    coordinates. Function stores the stress, strain and moment response to txt files.
    '''
    # Define two nodes at (0,0) for zero-length element
    ops.node(1001, 0.0, 0.0, 0.0)
    ops.node(1002, 0.0, 0.0, 0.0)

    # Fix all degrees of freedom except axial load and bending
    ops.fix(1001, 1, 1, 1, 1, 1, 1)
    ops.fix(1002, 0, 1, 1, 1, 0, 0)
    
    # Define element
    #                              tag ndI ndJ secTag
    ops.element('zeroLengthSection', 2001, 1001, 1002, parameters.SecTag)

    # Create recorder to extract the response 
    ops.recorder('Element', '-file', strains2[0], '-time', '-ele', 2001, 'section', 'fiber', dist1, -dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains2[1], '-time', '-ele', 2001, 'section', 'fiber', -dist1, dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains2[2], '-time', '-ele', 2001, 'section', 'fiber', dist1, dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains2[3], '-time', '-ele', 2001, 'section', 'fiber', -dist1, -dist2, 'stressStrain')

    # Define constant axial load
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 3001, 1)
    ops.load(1002, axialLoad, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Define first direction moment
    if dof==6:
        ops.timeSeries('Constant', 3)
        ops.pattern('Plain', 3003, 3)
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, - m, 0.0)


    # Define analysis parameters
    ops.integrator('LoadControl', 0, 1, 0, 0)
    ops.system('SparseGeneral', '-piv')
    ops.test('EnergyIncr', 1e-9, 10)
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.algorithm('Newton')
    ops.analysis('Static')

    # Do one analysis for constant axial load
    ops.analyze(1)

    ops.loadConst('-time', 0.0)

    # Define reference moment
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 3002, 2)
	
    if dof==6:
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) 
    else:
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) 

    # Compute curvature increment
    dK = maxK*parameters.mu / parameters.numIncr

    # Use displacement control at node 2 for section analysis
    ops.integrator('DisplacementControl', 1002, dof, dK, 1, dK, dK)

    # Do the section analysis
    ops.analyze(parameters.numIncr)
   
    # Clean cash after the analysis
    ops.wipeAnalysis()
    ops.remove('node', 1001), ops.remove('node', 1002), ops.remove('ele', 2001)
    ops.remove('sp', 1001, 1), ops.remove('sp', 1001, 2), ops.remove('sp', 1001, 3)
    ops.remove('sp', 1001, 4), ops.remove('sp', 1001, 5), ops.remove('sp', 1001, 6)
    ops.remove('sp', 1002, 1), ops.remove('sp', 1002, 2), ops.remove('sp', 1002, 3)
    ops.remove('sp', 1002, 4), ops.remove('sp', 1002, 5), ops.remove('sp', 1002, 6)
    ops.remove('timeSeries', 1), ops.remove('loadPattern', 3001), ops.remove('timeSeries', 2)
    ops.remove('loadPattern', 3002)
    
    if dof==6:
        ops.remove('timeSeries', 3)
        ops.remove('loadPattern', 3003)
    
    # Clean recorders
    ops.remove('recorders')
    

def returnDivisorsPair(n):
    ''' 
    This function finds all divisor pairs for rebars. 
    Input is selected number of rebars. Output is list of sublist with
    divisor pairs.
    '''
    listDivisors=[]
    # Note that this loop runs till square root
    i = 1
    while i <= math.sqrt(n):
        if (n % i == 0) :
            if (n / i == i) :
                listDivisors.append((i, i))
            else :
                listDivisors.append((i, n/i))
        i = i + 1
    return listDivisors

def grid_params(listDivisorPairs):
    ''' 
    This function randomly select a list of pairs
    to construct the grid for rebar placement. 
    The input is a list of sublists, where
    each list contains divisors of the selected number of rebars.
    Outputs are grid parameters: w_g and d_g
    '''
    gridDimsPair =  listDivisorPairs[random.randint(1, len(listDivisorPairs)-1)]
    randomGridDimAssign = random.sample(range(0, 2), 2)
    w_g = gridDimsPair[randomGridDimAssign[0]]
    d_g = gridDimsPair[randomGridDimAssign[1]]

    return w_g, d_g 
