# Provides ports to solve QCE using different algorithms
# Currently supports Genetic algorithms, Bayesian Optimization or any MIP solver
# Also provides QUBO form of the problem that can be used for any generic QUBO solver

# Instead of using CE calculator as provided by ICET, we vectorize the operations
# This is to enable much faster predictions as compared to the default calculator
# All the results for temporal acceleration are provided using vectorized cost functions
# This enables a fair comparison; otherwise with default CE model, the accelerations are artifically higher.

import numpy as np

def prepare_GA_BO(extracted_data):
    """
    Prepares cost-function for Genetic Algorithm (GA) and Bayesian Optimization (BO)
    Vectorize the operations to enable highly efficient parallel vector algebra
    """
    FirstOrder =  {'coefficients':[], 'x1':[], 's1':[]}
    SecondOrder = {'coefficients':[], 'x1':[], 'x2':[], 's1':[], 's2':[]}
    ThirdOrder =  {'coefficients':[], 'x1':[], 'x2':[], 'x3':[], 's1':[], 's2':[], 's3':[]}
    FourthOrder =  {'coefficients':[], 'x1':[], 'x2':[], 'x3':[], 'x4':[], 's1':[], 's2':[], 's3':[], 's4':[]}
    
    for term in extracted_data:
        
        if len(term[1]) == 1:
            FirstOrder['coefficients'].append(term[0])
            FirstOrder['x1'].append(term[1][0][1])
            FirstOrder['s1'].append(term[1][0][0])
        
        elif len(term[1]) == 2:
            SecondOrder['coefficients'].append(term[0])
            SecondOrder['x1'].append(term[1][0][1])
            SecondOrder['x2'].append(term[1][1][1])
            SecondOrder['s1'].append(term[1][0][0])
            SecondOrder['s2'].append(term[1][1][0])
        
        elif len(term[1]) == 3:
            ThirdOrder['coefficients'].append(term[0])
            ThirdOrder['x1'].append(term[1][0][1])
            ThirdOrder['x2'].append(term[1][1][1])
            ThirdOrder['x3'].append(term[1][2][1])
            ThirdOrder['s1'].append(term[1][0][0])
            ThirdOrder['s2'].append(term[1][1][0])
            ThirdOrder['s3'].append(term[1][2][0])
        
        elif len(term[1]) == 4:
            FourthOrder['coefficients'].append(term[0])
            FourthOrder['x1'].append(term[1][0][1])
            FourthOrder['x2'].append(term[1][1][1])
            FourthOrder['x3'].append(term[1][2][1])
            FourthOrder['x4'].append(term[1][3][1])
            FourthOrder['s1'].append(term[1][0][0])
            FourthOrder['s2'].append(term[1][1][0])
            FourthOrder['s3'].append(term[1][2][0])
            FourthOrder['s4'].append(term[1][3][0])
            
    for order in [FirstOrder, SecondOrder, ThirdOrder, FourthOrder]:
        for key in order.keys():
            order[key] = np.array(order[key])
            
    return FirstOrder, SecondOrder, ThirdOrder, FourthOrder
  
def cost_function_GA(Occupations, Interactions):
    """
    Returns negative of the energy given a state vector of element indices
    corresponding to list `elements`
    Negative is returned since we made DEAP work on maximizing the outcome
    
    Occupations: Array of size same as dimensions of lattice [0,1,2,1,2,....]
    Interactions: Tuple of (FirstOrder, SecondOrder, ThirdOrder, FourthOrder) interaction costs;
                  This is the result of ```prepare_GA``` function
    """
    Occupations = np.array(Occupations)
    Sum = 0.0
    # First Order Interactions
    if len(Interactions[0]['coefficients']) == 0:
        return [-1*Sum]
    ElementalIndexVector = Occupations[Interactions[0]['s1']]
    X = Interactions[0]['x1']
    Vector = X[np.arange(len(X)), ElementalIndexVector]
    Sum += np.sum(Interactions[0]['coefficients'] * Vector)
    
    # Second Order Interactions
    if len(Interactions[1]['coefficients']) == 0:
        return [-1*Sum]
    ElementalIndexVector1 = Occupations[Interactions[1]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[1]['s2']]
    X = Interactions[1]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[1]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    Sum += np.sum(Interactions[1]['coefficients'] * Vector1 * Vector2)
    
    # Third Order Interactions
    if len(Interactions[2]['coefficients']) == 0:
        return [-1*Sum]
    ElementalIndexVector1 = Occupations[Interactions[2]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[2]['s2']]
    ElementalIndexVector3 = Occupations[Interactions[2]['s3']]
    X = Interactions[2]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[2]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    X = Interactions[2]['x3']
    Vector3 = X[np.arange(len(X)), ElementalIndexVector3]    
    Sum += np.sum(Interactions[2]['coefficients'] * Vector1 * Vector2 * Vector3)
    
    # Fourth Order Interactions
    if len(Interactions[3]['coefficients']) == 0:
        return [-1*Sum]
    ElementalIndexVector1 = Occupations[Interactions[3]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[3]['s2']]
    ElementalIndexVector3 = Occupations[Interactions[3]['s3']]
    ElementalIndexVector4 = Occupations[Interactions[3]['s4']]
    X = Interactions[3]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[3]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    X = Interactions[3]['x3']
    Vector3 = X[np.arange(len(X)), ElementalIndexVector3]    
    X = Interactions[3]['x4']
    Vector4 = X[np.arange(len(X)), ElementalIndexVector4]    
    Sum += np.sum(Interactions[3]['coefficients'] * Vector1 * Vector2 * Vector3 * Vector4)
    
    return [-1*Sum]
  
def cost_function_BO(Occupations, Interactions):
    """
    Returns energy given a state vector of element indices
    corresponding to list `elements`
    
    Occupations: Array of size same as dimensions of lattice [0,1,2,1,2,....]
    Interactions: Tuple of (FirstOrder, SecondOrder, ThirdOrder, FourthOrder) interaction costs
                  This is the output of the ```prepare_GA``` function.
    """
    Occupations = np.array(Occupations)
    Sum = 0.0
    # First Order Interactions
    if len(Interactions[0]['coefficients']) == 0:
        return Sum
    ElementalIndexVector = Occupations[Interactions[0]['s1']]
    X = Interactions[0]['x1']
    Vector = X[np.arange(len(X)), ElementalIndexVector]
    Sum += np.sum(Interactions[0]['coefficients'] * Vector)
    
    # Second Order Interactions
    if len(Interactions[1]['coefficients']) == 0:
        return Sum
    ElementalIndexVector1 = Occupations[Interactions[1]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[1]['s2']]
    X = Interactions[1]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[1]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    Sum += np.sum(Interactions[1]['coefficients'] * Vector1 * Vector2)
    
    # Third Order Interactions
    if len(Interactions[2]['coefficients']) == 0:
        return Sum
    ElementalIndexVector1 = Occupations[Interactions[2]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[2]['s2']]
    ElementalIndexVector3 = Occupations[Interactions[2]['s3']]
    X = Interactions[2]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[2]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    X = Interactions[2]['x3']
    Vector3 = X[np.arange(len(X)), ElementalIndexVector3]    
    Sum += np.sum(Interactions[2]['coefficients'] * Vector1 * Vector2 * Vector3)
    
    # Fourth Order Interactions
    if len(Interactions[3]['coefficients']) == 0:
        return Sum
    ElementalIndexVector1 = Occupations[Interactions[3]['s1']]
    ElementalIndexVector2 = Occupations[Interactions[3]['s2']]
    ElementalIndexVector3 = Occupations[Interactions[3]['s3']]
    ElementalIndexVector4 = Occupations[Interactions[3]['s4']]
    X = Interactions[3]['x1']
    Vector1 = X[np.arange(len(X)), ElementalIndexVector1]
    X = Interactions[3]['x2']
    Vector2 = X[np.arange(len(X)), ElementalIndexVector2]
    X = Interactions[3]['x3']
    Vector3 = X[np.arange(len(X)), ElementalIndexVector3]    
    X = Interactions[3]['x4']
    Vector4 = X[np.arange(len(X)), ElementalIndexVector4]    
    Sum += np.sum(Interactions[3]['coefficients'] * Vector1 * Vector2 * Vector3 * Vector4)
    
    return Sum

def GetQUBO(cemodel_bp, replace_bp, onehot_bp, site_num, vars_per_site):
    """
    Returns the minimization problem as objective and penalty QUBOs
    cemodel_bp: binary quadratic polynomial representing the QCE 
    replace_bp: binary quadratic polynomial to account for higher order term reductions
    onehot_bp: binary quadratic polynomial to account for elemental encoding constraints
    site_num: Number of sites in the prototype lattice that are changing
    vars_per_site: Number of variables used to represent every site
    """
    float2int_coeff = 100000000
    coeffs = [1]*len(cemodel_bp)

    bit_values = []
    for s in range(site_num):
        site_bits = [0]*vars_per_site
        site_bits[np.random.randint(0,vars_per_site)]=1
        [bit_values.append(i) for i in site_bits] 

    coeffs = [1]*1
    objective_bp = sum([[{'c':e['c']*j*float2int_coeff,'p':e['p']} for e in i] for i,j in zip(cemodel_bp,coeffs)],[])
    penalty_bp = sum(replace_bp,[]) + onehot_bp[0]
    
    return objective_bp, penalty_bp
