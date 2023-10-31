# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:34:23 2023

@author: thorburh

Test script to run model code on a small instance, and check all the required constraints hold

"""

#FIrstly, import all required pacakges
import networkx as nx
from mailopt.NetworkBuilding.CreateSmallInstance import CreateSmallNetworkInstance
from mailopt.Deterministic.DetMCModelClass import DetMCModel

#Create the small network
SmallNetwork=CreateSmallNetworkInstance()
#print details on the small network
#Number of WAs
#Number of streams
#Number of tethered WAs/ID streams
#Print out work area specific information

#From this network, create the small network optimisation model

#Dictionary specifying types of variables in model. Values are gurobi codes
VTypes = {
    "X": "C", #Flow variables
    "Y": "C", #Worker number variables
}

SmallNetworkModel=DetMCModel(SmallNetwork,VTypes=VTypes,DelayFlowPen=1000000)

#Set the objectives and optimise
SmallNetworkModel.Solve_Lex_Objective_Manual(['DelayMailCost','MinMaxWorkerTimeShift'],Check=False)

#Extract the solution
SmallNetworkSol=SmallNetworkModel.extractSolution2()

#Print basic solution information here
#Demo functions from DetModelSolutionClass
#Maybe even plot



# #Check the tethering
#SmallNetwork.Tethered
ShiftTimes=SmallNetwork.Times
for (SecondWA, FirstWA, Shift) in SmallNetwork.Tethered:
    #Print relevant tethering information
    print("Initial WA is ",FirstWA,", dependent WA is ",SecondWA)
    #Extract required information from solution
    FirstWAVals=[SmallNetworkSol.Y[FirstWA,t] for t in ShiftTimes]
    SecondWAVals=[SmallNetworkSol.Y[SecondWA,t] for t in ShiftTimes]
    FirstWANonZeroTimes=[ShiftTimes[i] for i in range(len(FirstWAVals)) if FirstWAVals[i]>0]
    SecondWANonZeroTimes=[ShiftTimes[i] for i in range(len(SecondWAVals)) if SecondWAVals[i]>0]
    #If the First WA has workers staffed at any point, we need to check the tethering
    if len(FirstWANonZeroTimes)>0:
        #Find last time the WA is staffed
        FirstWAFinish=max(FirstWANonZeroTimes)
        #First first time (if any) the second WA is staffed
        if len(SecondWANonZeroTimes)>0:
            SecondWAStart=min(SecondWANonZeroTimes)
            #Print information
            print("First WA Finishes at time ",FirstWAFinish)
            print("Second WA Starts at time ",SecondWAStart)
            #Assert tethering holds
            assert FirstWAFinish < SecondWAStart
    else:
        #If WA1 not staffed, just check the second isn't either
        assert len(SecondWANonZeroTimes)==0
        print("Neither Tethered WA is staffed on that shift")
        




##Show that the ID mapping works
WAs=nx.get_node_attributes(SmallNetworkModel.Data.DG2,name='WS')

OriginWA='2cImp'#WANumber is 4
Dest1WA='2cManLS'#WANumber is 6
Dest2WA='Imp'#WANumbers is 7
OriginWANodes=[n for (n,atts) in SmallNetworkModel.Data.DG2.nodes.items() if atts['WS']==4]
Dest1WANodes=[n for (n,atts) in SmallNetworkModel.Data.DG2.nodes.items() if atts['WS']==6]
Dest2WANodes=[n for (n,atts) in SmallNetworkModel.Data.DG2.nodes.items() if atts['WS']==7]

#Get the correct edges for the flows in and the flows out
InOriginWAEdges=[(u,v) for ((u,v),atts) in SmallNetworkModel.Data.DG2.edges.items() if v in OriginWANodes and WAs[u]!=4]
OutEdges1=[(u,v) for (u,v) in SmallNetworkModel.Data.DG2.edges() if u in OriginWANodes and v in Dest1WANodes]
OutEdges2=[(u,v) for (u,v) in SmallNetworkModel.Data.DG2.edges() if u in OriginWANodes and v in Dest2WANodes]

#Get the flows along these edges, and ensure the indirect mapping is correct
InOriginTotalFlow=sum([SmallNetworkSol.X[u,v,k] for (u,v) in InOriginWAEdges for k in SmallNetworkModel.Data.Comods if (u,v,k) in SmallNetworkSol.X])
OutFlow1=[SmallNetworkSol.X[u,v,k] for (u,v) in OutEdges1 for k in SmallNetworkModel.Data.Comods if (u,v,k) in SmallNetworkSol.X]
OutFlow2=[SmallNetworkSol.X[u,v,k] for (u,v) in OutEdges2 for k in SmallNetworkModel.Data.Comods if (u,v,k) in SmallNetworkSol.X]
print("Total flow into ",OriginWA," is ",InOriginTotalFlow)
print("FLow into ", Dest1WA," is ", sum(OutFlow1))
print("Flow into ", Dest2WA," is ", sum(OutFlow2))

