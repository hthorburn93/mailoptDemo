# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:18:11 2020

@author: thorburh
"""


from datetime import datetime
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from .NetworkBuildingFunction import TimeExpand, AddSourceSink #,DG, B, MachineWS, Times,
from ..Graphing.GraphDrawing import CreateNodePos, PlotDiGraph#, UWS
#from NetworkBuilding import DG, Comods, B, MachineWS, Times



##CREATE NETWORK AND CAPACITIES


##Define the base network, and associated data
DG = nx.DiGraph()

DG.add_edges_from([('ImpA','1cImp'),('ImpA','1cManLS'),('ImpA','ManSeg'),('ImpA','2cManLS'),('ImpA','2cImp'),
                   ('1cImp','1cManLS'),
                   ('ManSeg','1cManLS'),('ManSeg','2cManLS'),
                   ('2cImp','2cManLS'),
                   ('1cManLS','Inw'),
                   ('2cManLS','Inw'),
                   ('Inw','Completion')])

plt.plot()
#<matplotlib.axes._subplots.AxesSubplot object at ...>
nx.draw(DG, with_labels=True, font_weight='bold')
plt.close()


##Set unique list of work stations

UWS=['Source', 'ImpA', '1cImp', '2cImp', 'ManSeg', '1cManLS',  '2cManLS', 'Inw', 'Completion', 'Sink']


##Set list of comodities

Comods=['1cMechLet',
        '1cSCMLet',
        '1cManLet',
        '2cMechLet',
        '2cSCMLet',
        '2cManLet',]

## Define the necessary data
# Comodities
#Comods=['1cPost','2cPost']
# Source/sink demands
B=[200 for i in range(len(Comods))]
# Discrete time poinds
Times=[0,1,2,3,4,5,6,7]
# Work areas run by machines
MachineWS=['ImpA', '1cImp', '2cImp']






##Get time-expanded network
DG2, NodeAtts, PlaceTimeDict = TimeExpand(DG, Times, Comods, B, MachineWS)

#print(list(DG2.edges()))
#And source and sink (with delay arcs)
Sources=[('Source',PlaceTimeDict[('ImpA',0)]) for i in range(len(DG.nodes()))]
Sinks=[('Sink',PlaceTimeDict[(i,Times[-1])]) for i in list(DG.nodes())]

Demands=[dict(zip(Comods,[1,1,1,1,1,1])) for i in range(len(DG.nodes()))]

AddSourceSink(DG2,Sources,Sinks,Demands)

#print("-------------")

#print(list(DG2.edges()))
#Get list of work stations for all nodes in DG2
WS=nx.get_node_attributes(DG2,name='WS')

#Get a list of all Delay Arc Indices
DelayArcInds=[i for i in range(len(DG2.edges())) if list(DG2.edges())[i][1]=='Sink' and WS[list(DG2.edges())[i][0]]!='Completion']

#Plot to make sure network looks correct
#PlotDiGraph(DG2,UWS)

ComCap={}
TotalCap={}


#Capacities are manual comodity-specific (K1), manual total (K2), mechanical Comodity specific (K3), mechanical total (K4)
K1=200
K2=400
K3=GRB.INFINITY
K4=GRB.INFINITY

for i in list(DG2.edges()):
    #If edge connects the same work area in two consecutive time periods, have infinite capacity for both comodities
        if WS[i[0]] == WS[i[1]]:
            ComCap[i]={j:GRB.INFINITY for j in Comods}#[GRB.INFINITY for j in range(len(Comods))]
            TotalCap[i]=GRB.INFINITY
        #Else, if edge is connected to a source or a sink, have infinite capacity for both comodities
        elif "Source" in str(WS[i[0]]) or "Sink" in str(WS[i[1]]):
            ComCap[i]={j:GRB.INFINITY for j in Comods}#[GRB.INFINITY for j in range(len(Comods))]
            TotalCap[i]=GRB.INFINITY
        #Else, set arcs coming from ImpA
        elif WS[i[0]]=='ImpA':
            #If it's going to 1cImp, it's a mechanically sorted arc for 1cMechLet only
            if WS[i[1]]=='1cImp':
                ComCap[i]=dict(zip(Comods,[K3, 0, 0, 0, 0, 0]))
                TotalCap[i]=K4
            #If it's going to 1cManLS, it's a mechanically sorted arc for 1cSCM only
            elif WS[i[1]]=='1cManLS':
                ComCap[i]=dict(zip(Comods,[0, K3, 0, 0, 0, 0]))
                TotalCap[i]=K4
            #If it's going to ManSeg, it's a mechanically sorted arc for 1cManLet and 2cManLet
            elif WS[i[1]]=='ManSeg':
                ComCap[i]=dict(zip(Comods,[0, 0, K3, 0, 0, K3]))
                TotalCap[i]=K4
            #If it's going to 2cManLS, it's a mechanically sorted arc for 2cSCM only
            elif WS[i[1]]=='2cManLS':
                ComCap[i]=dict(zip(Comods,[0, 0, 0, 0, K3, 0]))
                TotalCap[i]=K4
            #If it's going to 2cImp, it's a mechanically sorted arc for 2cMechLet only
            elif WS[i[1]]=='2cImp':
                ComCap[i]=dict(zip(Comods,[0, 0, 0, K3, 0, 0, 0]))
                TotalCap[i]=K4
        #Else, set arc going from 1cImp. Only arc goes to 1cManLS - mechanically sorted arc for 1cMechLet only
        elif WS[i[0]]=='1cImp':
            ComCap[i]=dict(zip(Comods,[K3, 0, 0, 0, 0, 0]))
            TotalCap[i]=K4
        #Else set arcs going from ManSeg
        elif WS[i[0]]=='ManSeg':
            #If it's going to 1cManLS, it's a manually sorted arc for 1cManLet only
            if WS[i[1]]=='1cManLS':
                ComCap[i]=dict(zip(Comods,[0, 0, K1, 0, 0, 0]))
                TotalCap[i]=K2
            #If it's going to 2cManLS, it's a manually sorted arc for 2cManLet only
            elif WS[i[1]]=='2cManLS':
                ComCap[i]=dict(zip(Comods,[0, 0, 0, 0, 0, K1]))
                TotalCap[i]=K2
            #If there's another arc, this indicates a problem
            else:
                print("Error: Where's this arc going?")
                print((WS[i[0]],WS[i[1]]))
        #Else, set arc going from 2cImp. Only arc goes to 1cManLS - mechanically sorted arc for 2cMechLet only
        elif WS[i[0]]=='2cImp':
            ComCap[i]=dict(zip(Comods,[0, 0, 0, K3, 0, 0]))
            TotalCap[i]=K4
        #Else, set arc going from 1cManLS. This is a manual arc for all 1c letters
        elif WS[i[0]]=='1cManLS':
            ComCap[i]=dict(zip(Comods,[K1, K1, K2, 0, 0, 0]))
            TotalCap[i]=K2
        #Else, set arc going from 2cManLS. This is a manual arc for all 2c letters
        elif WS[i[0]]=='2cManLS':
            ComCap[i]=dict(zip(Comods,[0, 0, 0, K1, K1, K2]))
            TotalCap[i]=K2
        #Else, set arc going from Inw. This is a manual arc for all letters.
        elif WS[i[0]]=='Inw':
            ComCap[i]={j:K1 for j in Comods}#[K1 for j in range(len(Comods))]
            TotalCap[i]=K2
        else:
            print("Error: weird arc", (WS[i[0]],WS[i[1]]))

        
