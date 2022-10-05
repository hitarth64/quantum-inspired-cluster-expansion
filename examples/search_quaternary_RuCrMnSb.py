from icet import ClusterSpace, StructureContainer, ClusterExpansion
from sklearn.linear_model import LassoCV, Lasso
from ase.io import read
import pandas as pd
import numpy as np
from QuantumInspiredClusterExpansion import data 
from sklearn.metrics import * 
from sklearn.model_selection import train_test_split, cross_val_predict
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
import QuantumInspiredClusterExpansion as qce
from QuantumInspiredClusterExpansion import utils, engines
import pickle

# Load the relevant dataset
# You can get more information about pre-available datasets using: DatasetInstance.GetInfo()
DatasetInstance = data.Dataset()
df = DatasetInstance.GetData('d3')

# Ordering of elements should be same in the ase atoms object for ICET to work
# Usually this is the case. If not, we paste a small code that can help you do that very easily
for i in df.index.values:
    positions = df.loc[i,'atoms'].positions
    sorted_ = np.lexsort((positions[:,2], positions[:,1], positions[:,0]))
    df.at[i,'atoms'] = df.loc[i,'atoms'][sorted_]

# Remove pure phases since there aren't many of them to learn from
df = df[df['MixingEnergy']!=0.0]

# Find list of allowed elements for every site
chemsymbols = df.loc[10,'atoms'].get_chemical_symbols()
allowed_elements = ["W","Cr","Mn","Ti","Co","Sb","V","Ru"]
allowed_Zs = [Element(e).Z for e in allowed_elements]
substitutions = []
for chem in chemsymbols:
    if chem != "O":
        substitutions.append(allowed_elements)
    else:
        substitutions.append(["O"])

# Preparing data for training CE models for mixing energies
cs_mixing = ClusterSpace(structure=df.loc[10,'atoms'], cutoffs=[4,8], chemical_symbols=substitutions)
sc_mixing = StructureContainer(cluster_space=cs_mixing)
for i in df.index.values:
    sc_mixing.add_structure(structure=df.loc[i,'atoms'],properties={'mixing_energy': df.loc[i,'MixingEnergy']/72, \
                                                            'pband':df.loc[i,'pband'],
                                                            'dband':df.loc[i,'dband']}) #convert to ev/atom
model = LassoCV(cv=10, random_state=42)
print('Start training for mixing energies=======')
y = sc_mixing.get_fit_data(key='mixing_energy')[1]
X = sc_mixing.get_fit_data(key='mixing_energy')[0]
model.fit(X, y)
ce_mixing = ClusterExpansion(cluster_space=cs_mixing, parameters=model.coef_) 
intercept_mixing = model.intercept_
crossmodel = Lasso(alpha=model.alpha_, fit_intercept=model.fit_intercept, normalize=model.normalize,\
                   max_iter=model.max_iter, tol=model.tol, positive=model.positive,\
                  random_state=model.random_state, selection=model.selection)
predicted = cross_val_predict(crossmodel, X, y, cv=10)
print('MAE:{:.3f}'.format(mean_absolute_error(predicted,y)))
print('R2:{:.3f}'.format(r2_score(y,predicted)))


# Preparing data for training CE models for p-band centers
cs_pavg = ClusterSpace(structure=df.loc[10,'atoms'], cutoffs=[4], chemical_symbols=substitutions)
sc_pavg = StructureContainer(cluster_space=cs_pavg)
for i in df.index.values:
    sc_pavg.add_structure(structure=df.loc[i,'atoms'],properties={'mixing_energy': df.loc[i,'MixingEnergy']/72, \
                                                            'pband':df.loc[i,'pband'],
                                                            'dband':df.loc[i,'dband']}) #convert to ev/atom
model = LassoCV(cv=10, random_state=42)
print('Start training for pband centers =======')
y = sc_pavg.get_fit_data(key='pband')[1]
X = sc_pavg.get_fit_data(key='pband')[0]
model.fit(X, y)
ce_pavg = ClusterExpansion(cluster_space=cs_pavg, parameters=model.coef_) 
intercept_pavg = model.intercept_
crossmodel = Lasso(alpha=model.alpha_, fit_intercept=model.fit_intercept, normalize=model.normalize,\
                   max_iter=model.max_iter, tol=model.tol, positive=model.positive,\
                  random_state=model.random_state, selection=model.selection)
predicted = cross_val_predict(crossmodel, X, y, cv=100)
print('MAE:{:.3f}'.format(mean_absolute_error(predicted,y)))
print('R2:{:.3f}'.format(r2_score(y,predicted)))

# Preparing data for training CE models for dband centers
cs_davg = ClusterSpace(structure=df.loc[10,'atoms'], cutoffs=[12,6], chemical_symbols=substitutions)
sc_davg = StructureContainer(cluster_space=cs_davg)
for i in df.index.values:
    sc_davg.add_structure(structure=df.loc[i,'atoms'],properties={'mixing_energy': df.loc[i,'MixingEnergy']/72, \
                                                            'pband':df.loc[i,'pband'],
                                                            'dband':df.loc[i,'dband']}) #convert to ev/atom
model = LassoCV(cv=10, random_state=42)
print('Start training for dband centers =======')
y = sc_davg.get_fit_data(key='dband')[1]
X = sc_davg.get_fit_data(key='dband')[0]
model.fit(X, y)
ce_davg = ClusterExpansion(cluster_space=cs_davg, parameters=model.coef_) 
intercept_davg = model.intercept_
crossmodel = Lasso(alpha=model.alpha_, fit_intercept=model.fit_intercept, normalize=model.normalize,\
                   max_iter=model.max_iter, tol=model.tol, positive=model.positive,\
                  random_state=model.random_state, selection=model.selection)
predicted = cross_val_predict(crossmodel, X, y, cv=10)
print('MAE:{:.3f}'.format(mean_absolute_error(predicted,y)))
print('R2:{:.3f}'.format(r2_score(y,predicted)))

# Converting CE model to QCE
record_mixing = qce.ICET2List(cs_mixing, ce_mixing, new_atom=df.loc[0,'atoms'])
utils.verify_record(record_mixing, ce_mixing, ce_mixing.parameters, intercept_mixing, df.loc[0,'atoms']) # last argument can be any valid atoms object

record_pavg = qce.ICET2List(cs_pavg, ce_pavg, new_atom=df.loc[0,'atoms'])
utils.verify_record(record_pavg, ce_pavg, ce_pavg.parameters, intercept_pavg, df.loc[0,'atoms'])

record_davg = qce.ICET2List(cs_davg, ce_davg, new_atom=df.loc[0,'atoms'])
utils.verify_record(record_davg, ce_davg, ce_davg.parameters, intercept_davg, df.loc[0,'atoms'])

elements = [44,25,24,51] # Ru,Mn,Cr,Sb

weight1, weight2 = 8000, 1 # weights for the mixing-energy & efficiencies we used
# The exact coefficients multiplied against weights can be seen in equation (7) of main text
extracted_data = utils.ExtractDataFromLists([record_mixing, record_pavg, record_davg], [weight1-weight2, weight2*0.0459798, weight2*(-0.097515)], elements)

"""
Additional conditions we imposed for the searches that led to prediction of RuCrMnO2 are listed below:
- %Ru >= 50%
- %Cr > 0%
- %Mn > 0%
""" 

"""
Additional conditions we imposed for the searches that led to prediction of RuCrMnSbO2 are listed below:
- %Ru >= 58%
- %Cr > 0%
- %Mn > 0%
- [%Sb > 0%] and [%Sb < 10%]. 
The upper bound was motivated by the fact that we only wanted to perturb the original composition a litle; thereby ensuring that synthesis doesn't fail
"""

# GA-based search
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import time

vectorizedinteractions = engines.prepare_GA_BO(extracted_data)
costfunctionGA = lambda x: engines.cost_function_GA(x, vectorizedinteractions)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 2) # Attribute generator 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 64) # Structure initializers
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", costfunctionGA)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
start = time.time()
pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=300, lambda_=400, cxpb=0.3, mutpb=0.7, ngen=100, stats=stats, halloffame=hof, verbose=True)
end = time.time()
print('Time taken (in seconds): ',end-start)

# BO-based search
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import time
from skopt import gp_minimize

costfunctionBO = lambda x: engines.cost_function_BO(x, vectorizedinteractions)
space = [Categorical([0,1,2])]*64
start = time.time()
res_gp = gp_minimize(costfunctionBO, space, n_calls=1000, random_state=0, verbose=2)
end = time.time()
print('Time taken (in seconds): ',end-start)

# QUBO-formulation
rt = {'table':{}, 'next_new_site_id':0}
model_num = 1
cemodel_bp = [None]*model_num
onehot_bp = [None]*model_num
replace_bp = [None]*model_num
vars_per_site = len(elements)

atom2site, site2atom, site_num = qce.mk_converter_from_atomid_to_siteid(extracted_data)
cemodel_bp[0], replace_bp[0], onehot_bp[0], rt = qce.generate_quboinput(extracted_data, atom2site, site_num, vars_per_site, rt=rt)
objective_bp, penalty_bp = engines.GetQUBO(cemodel_bp, replace_bp, onehot_bp, site_num, vars_per_site)
with open('objectives_penalties.pkl','wb') as f:
    pickle.dump([objective_bp, penalty_bp], f)

# These objective_bp and penalty_bp are json formatted QUBO expressions. These can be used in combination with a QUBO solver to solve the problem
# We used Fujitsu's Digital Annealer (DA) which is not available open-source. You can find more details about DA here: 
# https://www.fujitsu.com/jp/documents/digitalannealer/researcharticles/DA_WP_EN_20210922.pdf
