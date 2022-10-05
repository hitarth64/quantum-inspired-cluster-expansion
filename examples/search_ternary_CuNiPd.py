from icet import ClusterSpace, StructureContainer, ClusterExpansion
from sklearn.linear_model import LassoCV
from ase.io import read
import pandas as pd
import numpy as np
from QuantumInspiredClusterExpansion import data 
from sklearn.metrics import * 
from sklearn.model_selection import train_test_split
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
import QuantumInspiredClusterExpansion as qce
from QuantumInspiredClusterExpansion import utils, engines
import pickle

# Load the relevant dataset
# You can get more information about pre-available datasets using: DatasetInstance.GetInfo()
DatasetInstance = data.Dataset()
df = DatasetInstance.GetData('d1')

# Preparing data for training CE models
cs = ClusterSpace(structure=df.loc[155,'atoms'], cutoffs=[5,5], chemical_symbols=['Cu','Ni','Pd','Ag'])
sc = StructureContainer(cluster_space=cs)
for i in df.index.values:
    sc.add_structure(structure=df.loc[i,'atoms'],properties={'mixing_energy': df.loc[i,'MixingEnergy']})

# Training ML model using Lasso regression
model = LassoCV(cv=5, random_state=42)
y = sc.get_fit_data(key='mixing_energy')[1]
X = sc.get_fit_data(key='mixing_energy')[0]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
model.fit(x_train, y_train)
print('R2 score on test data: {:.3f}'.format(model.score(x_test, y_test)))
preds = model.predict(x_test)
print("MAE on test data: {:.3f}".format(mean_absolute_error(preds,y_test)))

# Define the cluster expansion
ce = ClusterExpansion(cluster_space=cs, parameters=model.coef_)

# Converting CE model to QCE
record = qce.ICET2List(cs, ce, new_atom=df.loc[0,'atoms'])
utils.verify_record(record, ce, ce.parameters, model.intercept_, df.loc[155,'atoms']) # last argument can be any valid atoms object
elements = [28,29,46] #Ni,Cu,Pd
extracted_data = utils.ExtractDataFromLists([record], [1.0], elements)

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
