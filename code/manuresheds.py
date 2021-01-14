#Date: 1/5/21
#Manure Sheds implementation

### Imports ###
import pyomo.environ as pyo
from pyomo.core import *
import pandas as pd
import os

### Inputs ###
solvername='glpk'
solverpath_exe='C:\\Users\\Rachel\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\glpk-4.65\\w64\\glpsol' #Rachel
#solverpath_exe='C:\\Users\\Glenn.Moglen\\AppData\\Local\\Continuum\\anaconda3\\Lib\\glpk-4.65\\w64\\glpsol'	#Glenn
datapath=".\\..\\data\\"


### Read in data ###
demand = pd.read_csv(datapath+"demand.csv", header=None, index_col=0, squeeze=True).to_dict()
print(demand)
print(demand["Montgomery"])

flatrate = pd.read_csv(datapath+"flatrate.csv", index_col=0).to_dict()
# print(flatrate)
# print(flatrate["Montgomery"]["Howard"])
# print(flatrate.keys())



### Sets ###
M = pyo.ConcreteModel()
#M.ITEMS = pyo.Set(initialize=v.keys())

### Variables ###
#M.x = pyo.Var(M.ITEMS, within=Binary)

### Constraints ###
#M.weight = pyo.Constraint(expr=sum(w[i]*M.x[i] for i in M.ITEMS) <= limit)

### Objective Function ###
# def obj_rule(M):
#     return sum(v[i]*M.x[i] for i in M.ITEMS)
# M.obj = Objective(rule=obj_rule, sense=maximize)


### Solve and Show Results ###
# solver = pyo.SolverFactory(solvername,executable=solverpath_exe)
# results = solver.solve(M)
# for i in M.ITEMS:
#      print(M.x[i].value)

# results.write()