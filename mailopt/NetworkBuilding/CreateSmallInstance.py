# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:39:47 2021

@author: thorburh

Function to create a small test network, and associated demands
"""


from datetime import datetime
import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from mailopt.NetworkBuilding.NetworkBuildingFunction import TimeExpand, AddSourceSink, AddSourceSink2 #,DG, B, MachineWS, Times,
from mailopt.data import ProblemData
#from ..Graphing.GraphDrawing import CreateNodePos, PlotDiGraph#, UWS
#from NetworkBuilding import DG, Commods, B, MachineWS, Times


def CreateSmallNetworkInstance(AddDelays=True):
    '''
    
    Function to create an instance of a small mail centre network

    Returns
    -------
    DG2 : TYPE. A networkx digraph object
        DESCRIPTION. The directed time-expanded network you wish to solve - An output of function TimeExpand
    Times: Range
        DESCRIPTION. The range of time periods to be considered.
    ComCap : Dictionary
        DESCRIPTION. A dictionary of the capacities for each different commodity on each edge of the network
    TotalCap : TYPE. Dictionary
        DESCRIPTION. A dictionary of the total capacity on each edge of the network
    Commods : TYPE. List of Strings
        DESCRIPTION. A list of the names of all the different types of commodities.

    '''
    

    ##CREATE NETWORK AND CAPACITIES
    
    
    ##Define the base network, and associated data
    DG = nx.DiGraph()
    
    DG.add_edges_from([(1,2),(1,5),(1,3),(1,6),(1,4),
                       (2,5),
                       (3,5),(3,6),
                       (4,6),(4,7),
                       (5,7),
                       (6,7),
                       (7,'Completion')])
    
# =============================================================================
#     plt.plot()
#     #<matplotlib.axes._subplots.AxesSubplot object at ...>
#     nx.draw(DG, with_labels=True, font_weight='bold')
#     plt.close()
# =============================================================================
    
    
    ##Set unique list of work stations
    
    #UWS=['Source', 1, 2, 4, 3, 5,  6, 7, 'Completion', 'Sink']
    UWS=[1, 2, 4, 3, 5,  6, 7, 'Completion']
    UWSNames=['WA_1','WA_2','WA_3','WA_4','WA_5','WA_6','WA_7','Completion']
    
    
    ##Set list of comodities
    
    Commods=['Commod_1',
            'Commod_2',
            'Commod_3',
            'Commod_4',
            'Commod_5',
            'Commod_6',]
    
    ## Define the necessary data
    # Comodities
    #Commods=['1cPost','2cPost']
    # Source/sink demands
    B=[200 for i in range(len(Commods))]
    # Discrete time poinds
    Times=[0,1,2,3,4,5,6,7,8]
    # Work areas run by machines
    MachineWS=[1, 2, 4]
    
    
    
    
    
    
    ##Get time-expanded network
    DG2, NodeAtts, PlaceTimeDict = TimeExpand(DG, Times, Commods, [], [])#TimeExpand(DG, Times, Commods, B, MachineWS)
    
    #print(list(DG2.edges()))
    #And source and sink (with delay arcs)
    #Sources=[('Source',PlaceTimeDict[(1,0)]) for i in range(len(DG.nodes()))]
    Sources2=[[[k,1,0,1]] for k in Commods]
    #Sinks=[('Sink',PlaceTimeDict[(i,Times[-1])]) for i in list(DG.nodes())]
    Sinks2=[[[k,'Completion',Times[-1],1]] for k in Commods]
    
    Demands=[dict(zip(Commods,[1000,1000,1000,1000,1000,1000])) for i in range(len(DG.nodes()))]
    
    #AddSourceSink(DG2,Sources,Sinks,Demands)
    AddSourceSink2(DG2,Sources2,Sinks2,Commods,PlaceTimeDict,Times[-1],AddDelays)
    
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
                ComCap[i]={j:GRB.INFINITY for j in Commods}#[GRB.INFINITY for j in range(len(Commods))]
                TotalCap[i]=GRB.INFINITY
            #Else, if edge is connected to a source or a sink, have infinite capacity for both comodities
            elif "Source" in str(WS[i[0]]) or "Sink" in str(WS[i[1]]):
                ComCap[i]={j:GRB.INFINITY for j in Commods}#[GRB.INFINITY for j in range(len(Commods))]
                TotalCap[i]=GRB.INFINITY
            #Else, set arcs coming from WA_1
            elif WS[i[0]]==1:
                #If it's going to WA_2, it's a mechanically sorted arc for Commod_1 only
                if WS[i[1]]==2:
                    ComCap[i]=dict(zip(Commods,[K3, 0, 0, 0, 0, 0]))
                    TotalCap[i]=K4
                #If it's going to WA_5, it's a mechanically sorted arc for 1cSCM only
                elif WS[i[1]]==5:
                    ComCap[i]=dict(zip(Commods,[0, K3, 0, 0, 0, 0]))
                    TotalCap[i]=K4
                #If it's going to WA_3, it's a mechanically sorted arc for Commod_3 and Commod_6
                elif WS[i[1]]==3:
                    ComCap[i]=dict(zip(Commods,[0, 0, K3, 0, 0, K3]))
                    TotalCap[i]=K4
                #If it's going to WA_6, it's a mechanically sorted arc for 2cSCM only
                elif WS[i[1]]==6:
                    ComCap[i]=dict(zip(Commods,[0, 0, 0, 0, K3, 0]))
                    TotalCap[i]=K4
                #If it's going to WA_4, it's a mechanically sorted arc for Commod_4 only
                elif WS[i[1]]==4:
                    ComCap[i]=dict(zip(Commods,[0, 0, 0, K3, 0, 0, 0]))
                    TotalCap[i]=K4
            #Else, set arc going from WA_2. Only arc goes to WA_5 - mechanically sorted arc for Commod_1 only
            elif WS[i[0]]==2:
                ComCap[i]=dict(zip(Commods,[K3, 0, 0, 0, 0, 0]))
                TotalCap[i]=K4
            #Else set arcs going from WA_3
            elif WS[i[0]]==3:
                #If it's going to WA_5, it's a manually sorted arc for Commod_3 only
                if WS[i[1]]==5:
                    ComCap[i]=dict(zip(Commods,[0, 0, K1, 0, 0, 0]))
                    TotalCap[i]=K2
                #If it's going to WA_6, it's a manually sorted arc for Commod_6 only
                elif WS[i[1]]==6:
                    ComCap[i]=dict(zip(Commods,[0, 0, 0, 0, 0, K1]))
                    TotalCap[i]=K2
                #If there's another arc, this indicates a problem
                else:
                    print("Error: Where's this arc going?")
                    print((WS[i[0]],WS[i[1]]))
            #Else, set arc going from WA_4. Only arc goes to WA_5 - mechanically sorted arc for Commod_4 only
            elif WS[i[0]]==4:
                ComCap[i]=dict(zip(Commods,[0, 0, 0, K3, 0, 0]))
                TotalCap[i]=K4
            #Else, set arc going from WA_5. This is a manual arc for all 1c letters
            elif WS[i[0]]==5:
                ComCap[i]=dict(zip(Commods,[K1, K1, K2, 0, 0, 0]))
                TotalCap[i]=K2
            #Else, set arc going from WA_6. This is a manual arc for all 2c letters
            elif WS[i[0]]==6:
                ComCap[i]=dict(zip(Commods,[0, 0, 0, K1, K1, K2]))
                TotalCap[i]=K2
            #Else, set arc going from WA_7. This is a manual arc for all letters.
            elif WS[i[0]]==7:
                ComCap[i]={j:K1 for j in Commods}#[K1 for j in range(len(Commods))]
                TotalCap[i]=K2
            else:
                print("Error: weird arc", (WS[i[0]],WS[i[1]]))
    WorkerCaps={(w,t):GRB.INFINITY for w in UWS for t in Times}
    C={w:1 for w in UWS}
    NodeCaps={n:K2 for n in DG2.nodes}
    #Create the stream paths
    StreamPaths={k:[] for k in Commods}
    AcceptPaths={k:[] for k in Commods}
    for k in Commods:
        for (orig,dest) in ComCap.keys():
            if WS[orig]!=WS[dest]:
                if ComCap[(orig,dest)][k]>0:
                    if (WS[orig],WS[dest]) not in AcceptPaths[k]:
                        if WS[dest]!='Sink':
                            AcceptPaths[k].append((WS[orig],WS[dest]))
    StreamPaths={'Commod_1':[1,2,5,7],
                 'Commod_2':[1,5,7],
                 'Commod_3':[1,3,5,7],
                 'Commod_4':[1,4,6,7],
                 'Commod_5':[1,6,7],
                 'Commod_6':[1,3,6,7],
                 }
    UWSLen=len(UWSNames)-1
    WANameNumber=dict(zip(UWSNames[:UWSLen],range(1,len(UWSNames))))
    WANumberName=dict(zip(range(1,len(UWSNames)),UWSNames[:UWSLen]))
    WorkPlanDict={"WA_Number":range(1,len(UWS)),
                  "WA_Name":UWSNames[:UWSLen],
                  "Early":["All" for i in range(len(UWS)-1)],
                  "Late":["All" for i in range(len(UWS)-1)],
                  "Night":["All" for i in range(len(UWS)-1)]}
    WorkPlan=pd.DataFrame(WorkPlanDict)
    #FINISH MAKING A ProblemData OBJECT HERE!!!!!!
    Shifts={"Early":range(3),
            "Late":range(3,6),
            "Night":range(6,9)}
    Tethered=[(6,3,"Night")]
    IDDicts={"WA_4":{"Origin":"WA_4",
                      "Destinations":["WA_6","WA_7"],
                      "Ratios":[0.05,0.95]}}
    #Add the Indirect stream paths to the AcceptPaths object
    #First, get all commodities going through the origin WA
    IDCommods=['Commod_4']
    for k in IDCommods:
        AcceptPaths[k].append((4,7))
    PD=ProblemData(DG2,Times,ComCap,WorkPlan,WorkerCaps,Commods,UWS,C,NodeCaps,DG,StreamPaths=StreamPaths,
                   AcceptPaths=AcceptPaths,WANameNumber=WANameNumber,WANumberName=WANumberName,
                   IDDicts=IDDicts,Tethered=Tethered,Shifts=Shifts)
    return PD