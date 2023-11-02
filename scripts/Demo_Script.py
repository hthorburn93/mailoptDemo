# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:34:23 2023

@author: thorburh

Test script to run model code on a small instance, and check all the required constraints hold

"""

#Firstly, import all required pacakges
import networkx as nx
from mailopt.NetworkBuilding.CreateSmallInstance import CreateSmallNetworkInstance
from mailopt.Deterministic.DetMCModelClass import DetMCModel

#Create the small network
SmallNetwork=CreateSmallNetworkInstance()
#print details on the small network
print("Given network instance consists of ")
print(len(SmallNetwork.MS)," WAs")
print(len(SmallNetwork.Comods)," streams")
print(len(SmallNetwork.Tethered)," tethered pairs")
print(len(SmallNetwork.IDDicts)," ID streams")
#Number of WAs
#Number of streams
#Number of tethered WAs/ID streams
#Print out work area specific information

#From this network, create the small network optimisation model

#Dictionary specifying types of variables in model. Values are gurobi codes
VTypes = {
    "X": "C", #Flow variables
    "Y": "I", #Worker number variables
}

SmallNetworkModel=DetMCModel(SmallNetwork,VTypes=VTypes,DelayFlowPen=1000000)

#Set the objectives and optimise
#Note - check is for
SmallNetworkModel.Solve_Lex_Objective_Manual(['DelayMailCost','MinMaxWorkerTimeShift'],Check=False)

#Extract the solution
SmallNetworkSol=SmallNetworkModel.extractSolution()

#Print basic solution information here
print("Model solved, with a status of ",SmallNetworkModel.Model.status)
print("Max workers is ",SmallNetworkSol.FindMax())
print("Changes is ",SmallNetworkSol.CountChangesShift())
print("Total delayed items is ",SmallNetworkSol.TotalArcCost(Penalty=1))
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

for o in SmallNetwork.IDDicts.values():
    OriginWA=o['Origin']
    Dests=o['Destinations']
    Ratios=o['Ratios']
    print("ID stream going from ",OriginWA,", with destinations:")
    print(Dests)
    print("and respective ratios")
    print(Ratios)
    assert len(Dests)==len(Ratios)
    OriginWANumber=SmallNetwork.WANameNumber[OriginWA]
    OriginWANodes=[n for (n,atts) in SmallNetworkModel.Data.DG2.nodes.items() if atts['WS']==OriginWANumber]
    InOriginWAEdges=[(u,v) for ((u,v),atts) in SmallNetworkModel.Data.DG2.edges.items() if v in OriginWANodes and WAs[u]!=OriginWANumber]
    InOriginTotalFlow=sum([SmallNetworkSol.X[u,v,k] for (u,v) in InOriginWAEdges for k in SmallNetworkModel.Data.Comods if (u,v,k) in SmallNetworkSol.X])
    print("Total flow into ",OriginWA," is ",InOriginTotalFlow)
    for i in range(len(Dests)):
        print("Ratio going to ",Dests[i],", is ",Ratios[i])
        DestWANumber=SmallNetwork.WANameNumber[Dests[i]]
        DestWANodes=[n for (n,atts) in SmallNetworkModel.Data.DG2.nodes.items() if atts['WS']==DestWANumber]
        OutEdges=[(u,v) for (u,v) in SmallNetworkModel.Data.DG2.edges() if u in OriginWANodes and v in DestWANodes]
        OutFlow1=[SmallNetworkSol.X[u,v,k] for (u,v) in OutEdges for k in SmallNetworkModel.Data.Comods if (u,v,k) in SmallNetworkSol.X]
        print("Flow into ", Dests[i]," is ", sum(OutFlow1))



