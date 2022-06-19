# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:40:03 2021

@author: Glenn.Moglen
"""
solvername='glpk'

solverpath_exe='C:\\Users\\Glenn.Moglen\\AppData\\Local\\Continuum\\anaconda3\\Lib\\glpk-4.65\\w64\\glpsol' #does not need to be directly on c drive

import pyomo.environ as pyo
from pyomo.core import *
import xlrd
import xlwt

# program starts here  

# this function creates a dictionary whose keys are strings stored in first column of spreadsheet
def readspread (book, sheetname):
#   sheet = book.sheet_by_index(sheetnum)   
   sheet = book.sheet_by_name(sheetname)
   nrows = sheet.nrows
   print ('spread', nrows, sheetname)
   key_list = []
   val_list = []
   for row_index in range(1, nrows):    # assumes first row is header row
      v1 = sheet.cell(row_index,0).value
      v2 = sheet.cell(row_index,1).value
      key_list.append(v1)
      val_list.append(v2)
   thezip = zip(key_list, val_list)
   my_dictionary = dict(thezip)
   return my_dictionary

# this function creates a dictionary whose keys are tuples
def readspread2 (book, sheetname):
   sheet = book.sheet_by_name(sheetname)   
   nrows = sheet.nrows
   print ('spread2', nrows)
   key_list = []
   val_list = []
   for row_index in range(1, nrows):    # assumes first row is header row
      c1 = sheet.cell(row_index,0).value
      c2 = sheet.cell(row_index,1).value
      v1 = sheet.cell(row_index,2).value
      x1 = [c1, c2]
      x1t = tuple(x1)
      key_list.append(x1t)
      val_list.append(v1)
   thezip = zip(key_list, val_list)
   my_dictionary = dict(thezip)
   return my_dictionary

# Start of Main Code
book = xlrd.open_workbook('C:\\ddrive\\manureshed_work\\test_dictionary.xlsx')
Demand = readspread(book, 'Demand')
T = readspread2(book, 'Distance_Matrix')
Pollute = readspread(book, 'Pollution_Factors')
Supply = readspread(book, 'Supply')

# Create an instance of the model
model = ConcreteModel()
model.dual = Suffix(direction=Suffix.IMPORT)

# Create objective function - w represents weight between two competing objectives
def obj_rule(model):
#    return 20*model.x1 + 10*model.x2
#    w = 0.10
    w = model.w
    costs_to_fertilize = sum([T[c,s]*model.x[c,s] for c in CUS for s in SRC])
    
# The block below attributes a water quality cost to excess nutrients left at the sources   
    psum = 0
    for s in SRC:
       ssum = 0
       for c in CUS:
           ssum = ssum + model.x[c,s]
       excess = Supply[s] - ssum
       psum = psum + excess * Pollute[s]
    costs_of_pollution = psum * 10
    return w * costs_to_fertilize + (1 - w) * costs_of_pollution

# Define index sets
CUS = list(Demand.keys())
SRC = list(Supply.keys())

# Define the decision variables "model.x"
model.x = Var(CUS, SRC, domain = NonNegativeReals)

# Constraints
model.src = ConstraintList()
for s in SRC:
    model.src.add(sum([model.x[c,s] for c in CUS]) <= Supply[s])
        
model.dmd = ConstraintList()
for c in CUS:
    model.dmd.add(sum([model.x[c,s] for s in SRC]) <= Demand[c])    # initially "=="
    
workbook = xlwt.Workbook()
sheet = workbook.add_sheet('optimize_results')  # this makes the sheet name "Hope"
sheet._cell_overwrite_ok = True
    
for iii in range(11):
    model.w = iii / 100   # model.w and w are the same thing. 
    w = iii / 100         # They are the weight applied on transportation cost
    
# Set the Objective function
    model.Cost = Objective(rule = obj_rule, sense = minimize)
    results = pyo.SolverFactory(solvername, executable = solverpath_exe)
    results.solve(model, tee=True)
    model.pprint()
    model.x.pprint()
    
    tsum = 0
    distsum = 0
    for c in CUS:
        csum = 0
        for s in SRC:
            print(c, s, model.x[c,s](),T[c,s])
            csum = csum + model.x[c,s]()
            tsum = tsum + model.x[c,s]()*T[c,s]
            if(model.x[c,s]() > 0):
               distsum = distsum + T[c,s]     
        print(csum, tsum, distsum)   
    
    psum = 0
    for s in SRC:
        ssum = 0
        for c in CUS:
            ssum = ssum + model.x[c,s]()
        excess = Supply[s] - ssum
        psum = psum + excess * Pollute[s]
        print ('Moved/Excess nutrients from/left at source:', ssum, excess)
        
    print (w, tsum, psum)
    sheet.write(iii, 1, w)
    sheet.write(iii, 2, tsum)
    sheet.write(iii, 3, psum)
    
workbook.save('C:\\ddrive\\manureshed_work\\optimization_results.xls')
