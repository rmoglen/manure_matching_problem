# Manure Shipment Problem
# Authors: Rachel Moglen and Katrina Maynor
# 6/19/22

#---------------------- IMPORTS ----------------------

import numpy as np
import pandas as pd
import networkx as nx
from mpl_toolkits.basemap import Basemap as Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyomo.environ as pyo
from pyomo.core import *
import time
import statistics
from plotting import visualize_inputs, plot_net, visualize_outputs

#---------------------- DATA and INPUTS ----------------------

rebuild=True
nutrient="N"    #"N" or "P"
state=["Arizona","California","Colorado","New Mexico","Nevada","Utah"]
data_dir="./../data/"
fig_dir="./../fig/"

supply_demand_filename="NuGIS_County_Data.csv"
county_distances_filename="sf12010countydistancemiles.csv"
county_coord_filename="county_locations.csv"

TjV=0.92    #$/ lb N     (price of synthetic fertilizer)
TiV=0.53	#$/ lb N     (cost of manure disposal)
TiF=141894  #$   		 (cost of manure disposal)
TjF=0  		#$			 (fixed cost of synthetic fertilizer)
trucking_rate=0.008995 	#$/ mi/ lb N
PiV=0  					#(cost of treating manure for crop application)
PjV=0.63   	#$/lb N  	 (cost of being able to apply manure)

#---------------------- MAIN ----------------------

t = time.time()
print("Rebuilding: ", str(rebuild))

## Build Network
if rebuild:
	# Read in data
	supply_demand = pd.read_csv(data_dir+ supply_demand_filename)
	county_coord = pd.read_csv(data_dir + county_coord_filename)
	county_distances = pd.read_csv(data_dir+ county_distances_filename)
	supply_demand=supply_demand.merge(county_coord[['FIPS','lat','lon']])
	supply_demand['coord'] = list(zip(supply_demand.lat, supply_demand.lon))
	if state is not None:   #select case study states
		supply_demand=supply_demand[supply_demand['state'].isin(state)]

	# Check for counties not in the supply_demand dataset
	# check if county1 not in supply_demand and drop edge if it isn't
	county_distances=county_distances.rename(columns={"county1": "FIPS"})
	county_distances=county_distances.merge(supply_demand[['FIPS','BalanceN_Tons','BalanceP2O5_Tons']])
	county_distances=county_distances.rename(columns={"FIPS": "county1", "county2": "FIPS",
						 'BalanceN_Tons': 'N_Balance_1', "BalanceP2O5_Tons":"P_Balance_1"})
	county_distances['N_Balance_1'].replace('', np.nan, inplace=True)
	county_distances.dropna(subset=['N_Balance_1'], inplace=True)

	# check if county2 not in supply_demand and drop edge if it isn't
	county_distances=county_distances.merge(supply_demand[['FIPS','BalanceN_Tons','BalanceP2O5_Tons']])
	county_distances=county_distances.rename(columns={"FIPS": "county2", 
						'BalanceN_Tons': 'N_Balance_2', "BalanceP2O5_Tons":"P_Balance_2"})
	county_distances['N_Balance_2'].replace('', np.nan, inplace=True)
	county_distances.dropna(subset=['N_Balance_2'], inplace=True)
	county_distances=county_distances.loc[(county_distances[nutrient+"_Balance_1"] >= 0) 
						& (county_distances[nutrient+"_Balance_2"] < 0)]

	# Build networkx network
	pos = pd.Series(supply_demand.coord.values,index=supply_demand.FIPS).to_dict()
	s_d_N = pd.Series(supply_demand.BalanceN_Tons.values,index=supply_demand.FIPS).to_dict()
	s_d_P = pd.Series(supply_demand.BalanceP2O5_Tons.values,index=supply_demand.FIPS).to_dict()

	# create networkx object for network
	G=nx.from_pandas_edgelist(county_distances, source="county1", target="county2", 
				edge_attr=["mi_to_county"], create_using=nx.DiGraph())
	nx.set_node_attributes(G,pos,"pos")
	nx.set_node_attributes(G,s_d_N,"N balance")
	nx.set_node_attributes(G,s_d_P,"P balance")

	nx.write_gpickle(G,data_dir+"network.gpickle")
else:
	G=nx.read_gpickle(data_dir+"network.gpickle")

visualize_inputs(G, fig_dir, nutrient)

## Build Pyomo Model
model = pyo.ConcreteModel()

# Sets
model.I_plus = pyo.Set(initialize=[k for k,v in 
						nx.get_node_attributes(G,nutrient+" balance").items() if v>=0])
model.I_minus = pyo.Set(initialize=[k for k,v in 
						nx.get_node_attributes(G,nutrient+" balance").items() if v<0])
model.I=model.I_plus.union(model.I_minus)

# Parameters
model.s = pyo.Param(model.I_plus, initialize={k:v for k,v in 
						nx.get_node_attributes(G,nutrient+" balance").items() if v>=0})
model.d = pyo.Param(model.I_minus, initialize={k:-v for k,v in 
						nx.get_node_attributes(G,nutrient+" balance").items() if v<0})
model.TjV=pyo.Param(initialize=TjV)
model.TiV=pyo.Param(initialize=TiV)
model.PjV=pyo.Param(initialize=PjV)
model.PiV=pyo.Param(initialize=PiV)
model.TjF=pyo.Param(initialize=TjF)
model.TiF=pyo.Param(initialize=TiF)

# Compute variable costs
cV={}
for i in model.I_plus:
	for j in model.I_minus:
		dist=county_distances.query('county1 =='+str(i)+ " and county2 =="+str(j))["mi_to_county"].values[0]
		cV[i,j]=trucking_rate*dist
model.cV=pyo.Param(model.I_plus, model.I_minus, initialize=cV)

# Variables
model.x=pyo.Var(model.I_plus, model.I_minus, bounds=(0,None)) 
model.z=pyo.Var(model.I, bounds=(0,None)) 
model.q=pyo.Var(model.I_plus, within=pyo.Binary) 

# Constraints
def source_mass_balance_rule(m, i):
	return sum(m.x[i, j] for j in m.I_minus) + m.z[i] == m.s[i]
model.source_mass_balance= pyo.Constraint(model.I_plus, rule=source_mass_balance_rule)

def sink_mass_balance_rule(m, j):
	return sum(m.x[i, j] for i in m.I_plus) + m.z[j] == m.d[j]
model.sink_mass_balance= pyo.Constraint(model.I_minus, rule=sink_mass_balance_rule)

def excess_supply_binary_rule(m, i):
	return m.z[i] <= m.s[i]*m.q[i]
model.excess_supply_binary= pyo.Constraint(model.I_plus, rule=excess_supply_binary_rule)

def obj_rule(m):
	return  (sum(sum(m.cV[i,j]*m.x[i,j] for j in m.I_minus) for i in m.I_plus) 
			+ sum(m.PjV*sum(m.x[i, j] for i in m.I_plus) for j in m.I_minus) 
			+ sum(m.PiV*sum(m.x[i, j] for j in m.I_minus) for i in m.I_plus) 
			+ sum(m.TiV*m.z[i] for i in m.I_plus)
			+ sum(m.TjV*m.z[i] for i in m.I_minus)
			+ sum (m.TiF*m.q[i] for i in m.I_plus))
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

model.write(data_dir+"model.lp")  #save LP model
opt = pyo.SolverFactory('gurobi')
results=opt.solve(model, tee=True) 
print("Elapsed Time:", (time.time() - t)/60,  "minutes")

## Format Output
visualize_outputs(model,fig_dir, G, nutrient)


