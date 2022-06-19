# Manure Transshipment Problem
# Authors: Rachel Moglen and Katrina Maynor
# 6/10/22

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

#---------------------- DATA and INPUTS ----------------------

rebuild=True
nutrient="N"    #"N" or "P"
state=["California","New Mexico", "Arizona", "Nevada", "Colorado", "Utah"]
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
PiF=0 			#(fixed cost of being able to treat manure for crop application)
PjF=0 			#(fixed cost of being able apply manure for crop application)
cF=0 			#(fixed cost of establishing manure shipment between a source and a sink)

#---------------------- FUNCTIONS ----------------------
def _compute_plotting_window(pos):
	"""
	args:
		pos (list): list of [x,y] coordinate tuples
	returns:
		ll (list): lower left [x,y] coordinate of window
		ur (list): upper right [x,y] coordinate of window
	"""
	# compute plotting window
	ll= [min([coord[1] for coord in pos]), min([coord[0] for coord in pos])]
	ur= [max([coord[1] for coord in pos]), max([coord[0] for coord in pos])]
	win_width=abs(ll[0]-ur[0])
	win_height=abs(ll[1]-ur[1])
	center=[(ll[0] + ur[0])/ 2, (ll[1] + ur[1])/ 2]

	if win_width<win_height:
		ll[0] = center[0] - 1.2*(win_height/2)
		ur[0] = center[0] + 1.2*(win_height/2)
		win_width=abs(ll[0]-ur[0])

	ll=[ll[0] - win_width*0.05, ll[1] - win_height*0.05]
	ur=[ur[0] + win_width*0.05, ur[1] + win_height*0.05]

	return ll, ur

def plot_net(G, fig_dir):
	"""
	args: 
		G (networkx object): networkx graph of network nodes
		fig_dir (str): relative path for output figures
	returns:
		none
	"""

	pos=dict(G.nodes(data='pos'))

	ll, ur = _compute_plotting_window(pos.values())

	# project node coordinates onto basemap
	m = Basemap(projection='merc',llcrnrlon=ll[0],llcrnrlat=ll[1],urcrnrlon=ur[0],
					urcrnrlat=ur[1], lat_ts=0, resolution='l',suppress_ticks=True)
	for nodeid, coord in pos.items():
		mx, my = m(coord[1], coord[0])
		pos[nodeid]=(mx,my)

	# plot network
	plt.figure()
	nx.draw_networkx_nodes(G,pos=pos,node_size=10,node_color='red',alpha=.5)
	# try:  #some case study networkx may not have edges
	# 	nx.draw_networkx_edges(G, pos=pos)
	# except:
	# 	pass
	plt.title(' Network', fontsize=15)

	#plot basemap
	m.drawstates()
	m.drawcoastlines()
	m.drawcountries()
	m.shadedrelief()
	m.drawparallels(np.arange(10,90,5),labels=[1,1,0,1]) 		# draw parallels
	m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])	# draw meridians

	plt.savefig(fig_dir+ "network")
	plt.close()

def plot_transshipment(model, G, fig_dir, nutrient):
	"""
	args: 
		model (pyomo object): solved model of transshipment
		G (networkx object): networkx graph of network nodes
		fig_dir (str): relative path for output figures
		nutrient (str): nutrient selected for case study
	returns:
		none
	"""

	edgelist=[]
	for edge in G.edges:
		if (G.nodes[edge[0]][nutrient+ " balance"] >=0 
			and G.nodes[edge[1]][nutrient+ " balance"] <0): #source to sink
			G.edges[edge]["optimal_transshipment"]=model.x[edge[0],edge[1]].value
		elif (G.nodes[edge[0]][nutrient+ " balance"] <0 
			and G.nodes[edge[1]][nutrient+ " balance"] >=0): #sink to source
			G.edges[edge]["optimal_transshipment"]=model.x[edge[1],edge[0]].value
		else: 
			G.edges[edge]["optimal_transshipment"]=0
		if G.edges[edge]["optimal_transshipment"]>=0.01:
			edgelist.append(edge)
	# print({k:v for k,v in nx.get_edge_attributes(G,"optimal_transshipment").items() if v>=1})

	#node color: source or sink
	for node in G.nodes:
		if G.nodes[node][nutrient+" balance"]>=0:
			G.nodes[node]["color"]="green"
		else:
			G.nodes[node]["color"]="red"

	pos=dict(G.nodes(data='pos'))

	ll, ur = _compute_plotting_window(pos.values())

	# project node coordinates onto basemap
	m = Basemap(projection='merc',llcrnrlon=ll[0],llcrnrlat=ll[1],urcrnrlon=ur[0],
					urcrnrlat=ur[1], lat_ts=0, resolution='l',suppress_ticks=True)
	for nodeid, coord in pos.items():
		mx, my = m(coord[1], coord[0])
		pos[nodeid]=(mx,my)

	# plot network
	plt.figure()
	
	plt.title('Optimal Transshipment Network', fontsize=15)

	#plot basemap
	m.drawstates()
	m.drawcoastlines()
	m.drawcountries()
	m.shadedrelief()
	m.drawparallels(np.arange(10,90,5),labels=[1,1,0,1]) 		# draw parallels
	m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])	# draw meridians

	# scale failure probability to colormap
	low, *_, high = sorted(nx.get_edge_attributes(G,"optimal_transshipment").values())
	norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
	mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.summer)
	nx.draw(G,pos=pos, arrows=True, edgelist=edgelist, width=1, 
		edge_color="black", node_size=15, 
		node_color=nx.get_node_attributes(G,"color").values())

	plt.tight_layout()
	plt.savefig(fig_dir+ "optimal_transshipment")
	plt.close()

def visualize_inputs(G, fig_dir, nutrient):
	"""
	args: 
		G (networkx object): networkx graph of network nodes
		fig_dir (str): relative path for output figures
		nutrient (str): nutrient selected for case study
	returns:
		none
	"""
	plt.hist(nx.get_node_attributes(G,nutrient+" balance").values())
	plt.title(nutrient+" Balance")
	plt.xlabel("Nutrient Balance (lbs)")
	plt.ylabel("Number of Counties")
	plt.savefig(fig_dir+ "nutrient_balance_hist")
	plt.close()

	plot_net(G, fig_dir)

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
	if state is not None:
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
	# print(county_distances.loc[(county_distances['county1'] == 48295) & (county_distances['county2'] == 48387)])
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
model.PjF=pyo.Param(initialize=PjF)
model.PiF=pyo.Param(initialize=PiF)
# model.cV = pyo.Param(model.I_plus, model.I_minus, initialize={k:v for k,v in 
# 						nx.get_edge_attributes(G,"mi_to_county").items()})

# Compute variable costs
cV={}
for i in model.I_plus:
	for j in model.I_minus:
		dist=county_distances.query('county1 =='+str(i)+ " and county2 =="+str(j))["mi_to_county"].values[0]
		cV[i,j]=trucking_rate*dist
model.cV=pyo.Param(model.I_plus, model.I_minus, initialize=cV)
# model.cF=pyo.Param(model.I_plus, model.I_minus, initialize=cF)

# Variables
model.x=pyo.Var(model.I_plus, model.I_minus, bounds=(0,None)) 
# model.y=pyo.Var(model.I_plus, model.I_minus, within=pyo.Binary) 
# model.w=pyo.Var(model.I, within=pyo.Binary) 
model.z=pyo.Var(model.I, bounds=(0,None)) 
# model.q=pyo.Var(model.I, within=pyo.Binary) 
model.q=pyo.Var(model.I_plus, within=pyo.Binary) 

# Constraints
def source_mass_balance_rule(m, i):
	return sum(m.x[i, j] for j in m.I_minus) + m.z[i] == m.s[i]
model.source_mass_balance= pyo.Constraint(model.I_plus, rule=source_mass_balance_rule)

def sink_mass_balance_rule(m, j):
	return sum(m.x[i, j] for i in m.I_plus) + m.z[j] == m.d[j]
model.sink_mass_balance= pyo.Constraint(model.I_minus, rule=sink_mass_balance_rule)

# def supply_binary_rule(m, i, j):
# 	return sum(m.x[i, j] for j in model.I_minus) <= m.s[i]*m.w[i]
# model.supply_binary= pyo.Constraint(model.I_plus, rule=supply_binary_rule)

# def demand_binary_rule(m, i, j):
# 	return sum(m.x[i, j] for i in model.I_plus) <= m.d[j]*m.w[j]
# model.demand_binary= pyo.Constraint(model.I_minus, rule=demand_binary_rule)

# def transport_binary_rule(m, i, j):
# 	return m.x[i, j] <= min(m.s[i], m.d[j])*m.y[i, j]
# model.transport_binary= pyo.Constraint(model.I_plus, model.I_minus, rule=transport_binary_rule)

def excess_supply_binary_rule(m, i):
	return m.z[i] <= m.s[i]*m.q[i]
model.excess_supply_binary= pyo.Constraint(model.I_plus, rule=excess_supply_binary_rule)

# def unmet_demand_binary_rule(m, j):
# 	return m.z[j] <= m.d[j]*m.q[j]
# model.unmet_demand_binary= pyo.Constraint(model.I_minus, rule=unmet_demand_binary_rule)

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
results=opt.solve(model, tee=False) 
print("Elapsed Time:", (time.time() - t)/60,  "minutes")

## Format Output

# for i in model.I_plus:
# 	for j in model.I_minus:
# 		print(str(i)+","+str(j)+":"+str(model.x[i,j].value))

synthetic_fertilizer=[]
for j in model.I_minus:
	synthetic_fertilizer.append(100*(model.z[j].value)/model.d[j])
plt.hist(synthetic_fertilizer)
plt.title("Percent Demand Met by Synthetic Fertilizer")
plt.xlabel("Percent Demand (%)")
plt.ylabel("Number of Counties")
plt.savefig(fig_dir+ "syth_fertilizer_hist")
plt.close()

manure_disposal=[]
for i in model.I_plus:
	manure_disposal.append(100*(model.z[i].value)/model.s[i])
plt.hist(manure_disposal)
plt.title("Percent Manure Disposed")
plt.xlabel("Percent Demand (%)")
plt.ylabel("Number of Counties")
plt.savefig(fig_dir+ "manure_disp_hist")
plt.close()

plot_transshipment(model, G, fig_dir, nutrient)