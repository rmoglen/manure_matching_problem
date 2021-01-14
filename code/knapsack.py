# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:35:41 2020

@author: Glenn.Moglen
"""

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Knapsack Problem
#

solvername='glpk'

solverpath_exe='C:\\Users\\Rachel\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\glpk-4.65\\w64\\glpsol' #does not need to be directly on c drive

import pyomo.environ as pyo
from pyomo.core import *

v = {'hammer':8, 'wrench':3, 'screwdriver':6, 'towel':11}
w = {'hammer':5, 'wrench':7, 'screwdriver':4, 'towel':3}

limit = 14

M = pyo.ConcreteModel()

M.ITEMS = pyo.Set(initialize=v.keys())

M.x = pyo.Var(M.ITEMS, within=Binary)

M.weight = pyo.Constraint(expr=sum(w[i]*M.x[i] for i in M.ITEMS) <= limit)

def obj_rule(M):
    return sum(v[i]*M.x[i] for i in M.ITEMS)

M.obj = Objective(rule=obj_rule, sense=maximize)

solver = pyo.SolverFactory(solvername,executable=solverpath_exe)
results = solver.solve(M)
for i in M.ITEMS:
     print(M.x[i].value)


results.write()