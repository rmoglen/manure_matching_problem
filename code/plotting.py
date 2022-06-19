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

def _plot_shipment(model, G, fig_dir, nutrient):
	"""
	args: 
		model (pyomo object): solved model of shipment
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
			G.edges[edge]["optimal_shipment"]=model.x[edge[0],edge[1]].value
		elif (G.nodes[edge[0]][nutrient+ " balance"] <0 
			and G.nodes[edge[1]][nutrient+ " balance"] >=0): #sink to source
			G.edges[edge]["optimal_shipment"]=model.x[edge[1],edge[0]].value
		else: 
			G.edges[edge]["optimal_shipment"]=0
		if G.edges[edge]["optimal_shipment"]>=0.01:
			edgelist.append(edge)
	# print({k:v for k,v in nx.get_edge_attributes(G,"optimal_shipment").items() if v>=1})

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
	
	plt.title('Optimal Manure Matching', fontsize=15)

	#plot basemap
	m.drawstates()
	m.drawcoastlines()
	m.drawcountries()
	m.shadedrelief()
	m.drawparallels(np.arange(10,90,5),labels=[1,1,0,1]) 		# draw parallels
	m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])	# draw meridians

	# scale failure probability to colormap
	low, *_, high = sorted(nx.get_edge_attributes(G,"optimal_shipment").values())
	norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
	mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.summer)
	nx.draw(G,pos=pos, arrows=True, edgelist=edgelist, width=1, 
		edge_color="black", node_size=15, 
		node_color=nx.get_node_attributes(G,"color").values())

	plt.tight_layout()
	plt.savefig(fig_dir+ "optimal_shipment")
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
	plt.hist(nx.get_node_attributes(G,nutrient+" balance").values(), color="c", edgecolor="w")
	plt.title(nutrient+" Balance")
	plt.xlabel("Nutrient Balance (lbs)")
	plt.ylabel("Number of Counties")
	plt.axvline(statistics.mean(nx.get_node_attributes(G,nutrient+" balance").values()), color='k', linewidth=1)
	plt.axvline(statistics.mean(nx.get_node_attributes(G,nutrient+" balance").values())+statistics.stdev(nx.get_node_attributes(G,nutrient+" balance").values()), color='grey', linestyle='dashed', linewidth=1)
	plt.axvline(statistics.mean(nx.get_node_attributes(G,nutrient+" balance").values())-statistics.stdev(nx.get_node_attributes(G,nutrient+" balance").values()), color='gray', linestyle='dashed', linewidth=1)
	plt.text(30000, 160, f'Mean: {statistics.mean(nx.get_node_attributes(G,nutrient+" balance").values()):.2f} lbs')
	plt.text(30000, 150, f'Std Dev: {statistics.stdev(nx.get_node_attributes(G,nutrient+" balance").values()):.2f} lbs')
	plt.savefig(fig_dir+ "nutrient_balance_hist")
	plt.close()

	plot_net(G, fig_dir)

def visualize_outputs(model, fig_dir, G, nutrient):
	"""
	args: 
		model (pyomo object): solved model of shipment
		fig_dir (str): relative path for output figures
		G (networkx object): networkx graph of network nodes
		nutrient (str): nutrient selected for case study
	returns:
		none
	"""
	_plot_shipment(model, G, fig_dir, nutrient)

	synthetic_fertilizer=[]
	for j in model.I_minus:
		synthetic_fertilizer.append(100*(model.z[j].value)/model.d[j])
	plt.hist(synthetic_fertilizer, color="c", edgecolor="w")
	plt.title("Percent of Demand Met by Synthetic Fertilizer")
	plt.xlabel("Percent Demand (%)")
	plt.ylabel("Number of Counties")
	plt.axvline(statistics.mean(synthetic_fertilizer), color='k', linewidth=1)
	plt.text(80, 110, f'Mean: {statistics.mean(synthetic_fertilizer):.2f}%')
	plt.text(80, 105, f'Std Dev: {statistics.stdev(synthetic_fertilizer):.2f}%')
	plt.savefig(fig_dir+ "syth_fertilizer_hist")
	plt.close()

	manure_disposal=[]
	for i in model.I_plus:
		manure_disposal.append(100*(model.z[i].value)/model.s[i])
	plt.hist(manure_disposal, color="c", edgecolor="w")
	plt.title("Percent of Manure Supply Disposed")
	plt.xlabel("Percent Supply (%)")
	plt.ylabel("Number of Counties")
	plt.axvline(statistics.mean(manure_disposal), color='k', linewidth=1)
	plt.text(80, 85, f'Mean: {statistics.mean(manure_disposal):.2f}%')
	plt.text(80, 80, f'Std Dev: {statistics.stdev(manure_disposal):.2f}%')
	plt.savefig(fig_dir+ "manure_disp_hist")
	plt.close()

	transport_distances=[]
	for i in model.I_plus:
		for j in model.I_minus:
			if model.x[i,j].value>0:
				transport_distances.append(G.edges[i,j]["mi_to_county"])
	plt.hist(transport_distances, color="c", edgecolor="w")
	plt.title("Manure Transportation Distances")
	plt.xlabel("Transportation Distance (miles)")
	plt.ylabel("Number of Shipments")
	plt.axvline(statistics.mean(transport_distances), color='k', linewidth=1)
	plt.axvline(statistics.mean(transport_distances)+statistics.stdev(transport_distances), color='grey', linestyle='dashed', linewidth=1)
	plt.axvline(statistics.mean(transport_distances)-statistics.stdev(transport_distances), color='gray', linestyle='dashed', linewidth=1)
	plt.text(500, 55, f'Mean: {statistics.mean(transport_distances):.2f}')
	plt.text(500, 51, f'Std Dev: {statistics.stdev(transport_distances):.2f}')
	plt.savefig(fig_dir+ "transport_dist_hist")
	plt.close()