# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:58:30 2020

@author: thorburh
"""



##Building a network flow model from Gurobi, now including capacities along delay arcs


from datetime import datetime
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from NetworkBuilding.NetworkBuildingFunction import TimeExpand, AddSourceSink
# from GraphDrawing import CreateNodePos, PlotDiGraph, UWS, CalcEdgeWidths
#from NetworkBuilding import DG, Comods, B, MachineWS, Times


if __name__ == '__main__':
    ##Create base network DG
    DG = nx.DiGraph()
    
    DG.add_edges_from([('ImpA','1cImp'),('ImpA','1cManLS'),('ImpA','ManSeg'),('ImpA','2cManLS'),('ImpA','2cImp'),
                   ('1cImp','1cManLS'),
                   ('ManSeg','1cManLS'),('ManSeg','2cManLS'),
                   ('2cImp','2cManLS'),
                   ('1cManLS','Inw'),
                   ('2cManLS','Inw'),
                   ('Inw','Completion')])
    
    ## Define the necessary data
    # Comodities
    Comods=['1cPost','2cPost']
    # Source/sink demands
    B=[200 for i in range(len(Comods))]
    # Discrete time poinds
    Times=[0,1,2,3,4,5,6,7]
    # Work areas run by machines
    MachineWS=['ImpA', '1cImp', '2cImp']
    
    ##Get time-expanded network
    DG2, NodeAtts, PlaceTimeDict = TimeExpand(DG, Times, Comods, B, MachineWS)
    
    Sources=[('Source',PlaceTimeDict[('ImpA',0)]) for i in range(len(DG.nodes()))]
    Sinks=[('Sink',PlaceTimeDict[(i,Times[-1])]) for i in list(DG.nodes())]
    
    
    AddSourceSink(DG2,Sources,Sinks,[[1,1] for i in range(len(DG.nodes()))])
    #AddSourceSink(DG2,[('TestSource',0)],[('TestSink',63)],[[1,1]])
    
    #PlotDiGraph(DG2,UWS)
    
    #Create a list for which edges are from manual work ares
    ManEdges=[i for i in list(DG2.edges(data='Type')) if i[2]=='Man']
    ManEdgeInds=[i for i in range(len(DG2.edges(data='Type'))) if list(DG2.edges(data='Type'))[i][2]=='Man']
    
    #Set commodity and total capacity for each edge
    # =============================================================================
    # ComCap={};
    # TotalCap={};
    # for i in list(DG2.edges()):
    #     ComCap[i]=[200,200]
    #     TotalCap[i]=400
    # =============================================================================
        
    Nodes1c=['1cImp','1cManLS']
    Nodes2c=['2cImp','2cManLS']   
    ComCap={};
    TotalCap={};
    #K1 is capacity per worker, K2 is total capacity
    K1=200
    K2=400
    for i in list(DG2.edges()):
        if nx.get_node_attributes(DG2,name='WS')[i[0]] == nx.get_node_attributes(DG2,name='WS')[i[1]]:
            ComCap[i]=[GRB.INFINITY, GRB.INFINITY]
            TotalCap[i]=GRB.INFINITY
        elif "Source" in str(nx.get_node_attributes(DG2,name='WS')[i[0]]) or "Sink" in str(nx.get_node_attributes(DG2,name='WS')[i[1]]):
            ComCap[i]=[GRB.INFINITY, GRB.INFINITY]
            TotalCap[i]=GRB.INFINITY
        elif nx.get_node_attributes(DG2,name='WS')[i[0]] in Nodes1c:# or nx.get_node_attributes(DG2,name='WS')[i[1]] in Nodes1c:
            ComCap[i]=[K1,0]#,20,20,20,20]
            TotalCap[i]=K2
        elif nx.get_node_attributes(DG2,name='WS')[i[1]] in Nodes1c:
            ComCap[i]=[GRB.INFINITY,0]#,20,20,20,20]
            TotalCap[i]=GRB.INFINITY
        elif nx.get_node_attributes(DG2,name='WS')[i[0]] in Nodes2c:# or nx.get_node_attributes(DG2,name='WS')[i[1]] in Nodes2c:
            ComCap[i]=[0,K1]
            TotalCap[i]=K2
        elif nx.get_node_attributes(DG2,name='WS')[i[1]] in Nodes2c:
            ComCap[i]=[0,GRB.INFINITY]
            TotalCap[i]=GRB.INFINITY
        else:
            #ComCap[i]=[K1,K1]
            #TotalCap[i]=K2
            ComCap[i]=[GRB.INFINITY, GRB.INFINITY]
            TotalCap[i]=GRB.INFINITY


def CCMSGDetNetworkFlowOpt(DG2,ComCap,TotalCap,Comods,MS,C,ZDist,
                           Alpha=None,ComodGroups=None,Z=1000, Y=None,Threads=1,Pairs=True, VTypes={'X':'I','Y':'I'}, Opt=True):
    """
    

    Parameters
    ----------
    DG2 : TYPE. A networkx digraph object
        DESCRIPTION. The directed time-expanded network you wish to solve - An output of function TimeExpand
    ComCap : Dictionary
        DESCRIPTION. A dictionary of the capacities for each different commodity on each edge of the network. Each entry in the dictionary is a dictionary itself, with Commodities as keys and capacities as values
    TotalCap : TYPE. Dictionary
        DESCRIPTION. A dictionary of the total capacity on each edge of the network
    Comods : TYPE. List of Strings
        DESCRIPTION. A list of the names of all the different types of commodities
    MS : TYPE. A list of strings
        DESCRIPTION. A list giving which work areas are manual work areas instead of mechanical
    C : TYPE A list of floats/integers OR (if Y is not 'None') A dictionary
        DESCRIPTION. If 'Y==None', A list giving the cost of one worker for each area of MS
                        If Y!=None, a dictionary, indexed by the delay arcs and commodity indices, giving the cost of 1 unit of delayed flow along each delay arc, for each delay commodity
    ZDist : TYPE, Function which takes an argument "size", which is a 2-integer list
        DESCRIPTION. The function to generate the random demand. The default is npr.normal.
    Alpha : TYPE, A list of floats
        DESCRIPTION. Amount of flow for each commodity that needs to be sorted on time (that is, from the completion node). The default is [0.95 for i in range(len(Comods))].
    ComodGroups : TYPE, Dictionary, optional
        DESCRIPTION. A dictionary giving which commodities belong to the same group - that is, share the same commodities specific capacity on an arc (i.e. letters from the same stream, but arriving at different times).
        The keys are the different groups, the values are lists of the commodities in that group. The default (if None given) is {i:[k] for (i,k) in enumerate(Comods)} (all types are unique).
    Z : TYPE, Integer
        DESCRIPTION. The number of different scenarios to simulate. The default is 1000.
    Y : TYPE. Either 'None' or a dictionary of the same length as MS
        DESCRIPTION. If 'None', indicates that the model needs to minimise the number of workers required.
                        Otherwise, this is a dictionary of the same length as MS (with the same keys) indicating how many workers are in each worker area
    Pairs : TYPE, Boolean
            DESCRIPTION. Boolean describing if sources and sinks are given as corresponding pairs, or independently. Default is True
    VTypes : TYPE, Dictionary
        DESCRIPTION. Dictionary giving the types of the X and Y variables. Keys are 'X' and 'Y'. Values are any string value that can be accepted as a gurobi variable type in m.AddVar. Default is {'X':'I','Y':'I'}
    


    Returns
    -------
    X : TYPE: A gurobipy tupledict
        DESCRIPTION: A dictionary of the flow variables in the solution
    Y : TYPE: A gurobipy tupledict
        DESCRIPTION: A dictionary of the number of workers for each work area.
    m : TYPE: A gurobipy model
        DESCRIPTION: The gurobi model created and optimised

    """
    #Get dictionary of work stations for each node
    WS=nx.get_node_attributes(DG2,name='WS')
    
    #Get indices for delay arcs
    #DelayArcInds=[i for i in range(len(DG2.edges())) if list(DG2.edges())[i][1]=='Sink' and WS[list(DG2.edges())[i][0]]!='Completion']
    DelayArcs=[i for i in DG2.edges() if WS[i[1]]=='Sink' and DG2.nodes(data='WS')[i[0]]!='Completion']
    #Create Alpha and ComodGroups if not specified
    if Alpha==None:
        Alpha={k:0.95 for k in Comods}#[0.95 for i in range(len(Comods))]
    if ComodGroups==None:
        ComodGroups={k: i for (i, k) in enumerate(Comods)}
    
    ##Checks and assertions#
    Demands = nx.get_node_attributes(DG2,name='Demand')

    if Y==None:
        assert len(C) == len(MS)
    elif Y!=None:
        assert len(C) == len(DelayArcs)*len(Comods)
    
    #Check all demands are the length of the comodities
    #Demands = list(nx.get_node_attributes(DG2,name='Demand').values())
    assert all([len(Demands[i]) == len(Comods) for i in Demands])
    
    ##Check that ComodGroups and Alpha are the same length as the comodities
    #Assertion 1
    assert len(Comods) == len(Alpha)
    #Assertion 2
    assert len(Comods) == len(ComodGroups)
    
    
    ##Setting up gurobi model
    #Create model object
    m=gp.Model(name="Basic")
    
    

    
    
    #Set up graph and demands for multiple scenarios
    
    #Generate demand in different scenarios
    #Depends on if sources/sinks have been added as related pairs, or added seperately
    #This is indicated with the 'Pairs' argument
    print("Starting Scenario generation")
    if Pairs==True:
        ZDemands=np.round(ZDist(size=[Z,len(Comods)]))
    elif Pairs==False:
        ZDemands, CVs = ZDist(size=[Z,len(Comods)])
        ZDemands=[dict(zip(Comods,z)) for z in ZDemands]
    else:
        assert Pairs in [True,False]
        
    
    ##Add sources and sinks to graph
    
    #Get list of nodes which are sources and sinks
    Sources=[i[0] for i in DG2.nodes(data='WS') if i[1]=='Source']
    Sinks=[i[0] for i in DG2.nodes(data='WS') if i[1]=='Sink']
    
    
    
    
    #Change demand at the source and sink nodes
    print("Changing source/sink demand")
    if Pairs==True:
        for i in Sources:
            Demands[i]=[dict(zip(Comods,j)) for j in ZDemands]
        
        for i in Sinks:
            Demands[i]=[dict(zip(Comods,-j)) for j in ZDemands]
    else:
        for i in Sources:
            Demands[i]=[dict(zip(Comods,[Demands[i][k]*(1+CVs[k]*z[k]) for k in Comods])) for z in ZDemands]
        #In this situation, have to assume there's only one sink, and that meeting deadlines is managed by constraining arc capacities
        #Sinks will be a list of 1, with the sink in it.
        for i in Sinks:
            Demands[i]=[{k:-1*sum([Demands[i][z][k] for i in Sources]) for k in Comods} for z in range(Z)]
        
        
        
    ##Add variables
    print("Adding variables...")
    #Add X variables - X[i,k] is how much of commodity k to send along edge i
    X=m.addVars(DG2.edges,Comods,Z,name="Flow",vtype=VTypes['X'])
    #Set a variable for each edge starting at a manual work area, which controls the capacity of that edge
    
    if Y == None:
        Y=m.addVars(MS,name="Cap",vtype=VTypes['Y'])
        Obj="Workers"
    else:
        Obj="DelayFlow"
    
    print("Adding constraints...")
    ##Add the Mass Balance Constraints
    #Finds the edges leading into and out-of each node, and matches the sum of the flow out- sum of the flow in to the demand for each commodity at each node
    for (n,attr) in DG2.nodes.items():
        if n not in WS:
            print("n is ", n, "Type is ",type(n),"In WS?", n in WS)
        if WS[n] in ['Source','Sink']:
            [[m.addConstr(gp.quicksum(X[u,v,k,z] for (u,v) in DG2.out_edges(n))-gp.quicksum(X[u,v,k,z] for (u,v) in DG2.in_edges(n)) ==Demands[n][z][k], name='MassBal') for k in Comods] for z in range(Z)]
        else:
            for k in Comods:
                for z in range(Z):
                    m.addConstr(gp.quicksum(X[u,v,k,z] for (u,v) in DG2.out_edges(n))-gp.quicksum(X[u,v,k,z] for (u,v) in DG2.in_edges(n)) == Demands[n][k], name='MassBal')
        
    
    ##Arc Capacity Constraints
    #NOTE: X is indexed by edge indices, but TotalCap and ComCap are indexed by edge names - hence the differences in the notation.
    #If the Edge is NOT a manual edge (that is, originating under a manual node), total flow has to be less than the total capacity
                    #Furthermore, total flow of all comodities in the same group (e.g. 1cLetters, etc) across all different sources must be less than the Commodity Capacity for that group
    for (e, e_attrs) in DG2.edges.items():
        u,v=e
        if e_attrs.get('Type') != 'Man':
            [m.addConstr(gp.quicksum(X[u,v,k,z] for k in Comods)<=TotalCap[e]) for z in range(Z)]
            for j in ComodGroups:
                [[m.addConstr(gp.quicksum(X[u,v,k,z] for k in ComodGroups[j])<=ComCap[e][k2]) for z in range(Z)] for k2 in ComodGroups[j]]
        #Otherwise if edge is manual, capacities must be multiplied by the Y variable (number of workers) in the relevant work area (that of the origin of the edge)
        else:
            #Find the workstation of the node
            WhichWS=DG2.nodes[e[0]]['WS']
            #Add the total Capacity Constraint.
            [m.addConstr(gp.quicksum(X[u,v,k,z] for k in Comods)<=TotalCap[e]*Y[WhichWS]) for z in range(Z)]
            #Add the comodity capacity constraints
            for j in ComodGroups:
                [[m.addConstr(gp.quicksum(X[u,v,k,z] for k in ComodGroups[j])<=Y[WhichWS]*ComCap[e][k2]) for z in range(Z)] for k2 in ComodGroups[j]]
    
    ##Finally, add the 'Chance Constraints' - the limit on flow sent along the delay arcs
    if Obj=="Workers":
        for j in Sinks:
            [m.addConstr(gp.quicksum(X[u,v,k,z] for (u,v) in DelayArcs for z in range(Z))<=(1-Alpha[k])*gp.quicksum(-1*Demands[j][z][k] for z in range(Z))) for k in Comods]
    
    
    ##Set objetive function
    if Obj=='Workers':
        m.setObjective(gp.quicksum(C[i]*Y[i] for i in MS), GRB.MINIMIZE)
    elif Obj=='DelayFlow':
        m.setObjective(gp.quicksum(C[i,k]*X[u,v,k,z] for (u,v) in DelayArcs for k in Comods for z in range(Z)), GRB.MINIMIZE)
    else:
        assert 1==2
    
    
    ##Optimise
    if Opt==True:
        print("Beginning optimisation")
        m.setParam("OutputFlag",1)
        m.setParam("Threads",Threads)
        m.optimize()
        
    return X,Y,m,Demands#,AlphaCons

if __name__ == '__main__':
    MS=['ManSeg', '1cManLS', '2cManLS', 'Inw']
    C={i:1 for i in MS}


def TSDetNetworkFlowOpt(DG2,ComCap,TotalCap,Comods,MS,C,d,ZDist,
                           Alpha=None,ComodGroups=None,Z=1000, Y=None,Threads=1,Pairs=True, VTypes={'X':'I','Y':'I'},Opt=True):
    """
    

    Parameters
    ----------
    DG2 : TYPE. A networkx digraph object
        DESCRIPTION. The directed time-expanded network you wish to solve - An output of function TimeExpand
    ComCap : Dictionary
        DESCRIPTION. A dictionary of the capacities for each different commodity on each edge of the network. Each entry in the dictionary is a dictionary itself, with Commodities as keys and capacities as values
    TotalCap : TYPE. Dictionary
        DESCRIPTION. A dictionary of the total capacity on each edge of the network
    Comods : TYPE. List of Strings
        DESCRIPTION. A list of the names of all the different types of commodities
    MS : TYPE. A list of strings
        DESCRIPTION. A list giving which work areas are manual work areas instead of mechanical
    C : TYPE A list of floats/integers OR (if Y is not 'None') A dictionary
        DESCRIPTION. If 'Y==None', A list giving the cost of one worker for each area of MS
                        If Y!=None, a dictionary, indexed by the delay arcs and commodity indices, giving the cost of 1 unit of delayed flow along each delay arc, for each delay commodity
    d : TYPE. A dictionary of dictionaries of floats
        DESCRIPTION. A dictonary - keys are delay arcs, values are dictionarys. The keys of the sub-dictionaries are commodities, the values are the delay cost for that commodity on that arc.
    ZDist : TYPE, Function which takes an argument "size", which is a 2-integer list
        DESCRIPTION. The function to generate the random demand. The default is npr.normal.
    Alpha : TYPE, A list of floats
        DESCRIPTION. Amount of flow for each commodity that needs to be sorted on time (that is, from the completion node). The default is [0.95 for i in range(len(Comods))].
    ComodGroups : TYPE, Dictionary, optional
        DESCRIPTION. A dictionary giving which commodities belong to the same group - that is, share the same commodities specific capacity on an arc (i.e. letters from the same stream, but arriving at different times).
        The keys are the different groups, the values are lists of the commodities in that group. The default (if None given) is {i:[k] for (i,k) in enumerate(Comods)} (all types are unique).
    Z : TYPE, Integer
        DESCRIPTION. The number of different scenarios to simulate. The default is 1000.
    Y : TYPE. Either 'None' or a dictionary of the same length as MS
        DESCRIPTION. If 'None', indicates that the model needs to minimise the number of workers required.
                        Otherwise, this is a dictionary of the same length as MS (with the same keys) indicating how many workers are in each worker area
    Pairs : TYPE, Boolean
            DESCRIPTION. Boolean describing if sources and sinks are given as corresponding pairs, or independently. Default is True
    VTypes : TYPE, Dictionary
        DESCRIPTION. Dictionary giving the types of the X and Y variables. Keys are 'X' and 'Y'. Values are any string value that can be accepted as a gurobi variable type in m.AddVar. Default is {'X':'I','Y':'I'}
    


    Returns
    -------
    X : TYPE: A gurobipy tupledict
        DESCRIPTION: A dictionary of the flow variables in the solution
    Y : TYPE: A gurobipy tupledict
        DESCRIPTION: A dictionary of the number of workers for each work area.
    m : TYPE: A gurobipy model
        DESCRIPTION: The gurobi model created and optimised

    """
    #Get dictionary of work stations for each node
    WS=nx.get_node_attributes(DG2,name='WS')
    
    #Get indices for delay arcs
    #DelayArcInds=[i for i in range(len(DG2.edges())) if list(DG2.edges())[i][1]=='Sink' and WS[list(DG2.edges())[i][0]]!='Completion']
    DelayArcs=[i for i in DG2.edges() if WS[i[1]]=='Sink' and DG2.nodes(data='WS')[i[0]]!='Completion']
    #Create Alpha and ComodGroups if not specified
    if Alpha==None:
        Alpha={k:0.95 for k in Comods}#[0.95 for i in range(len(Comods))]
    if ComodGroups==None:
        ComodGroups={k: i for (i, k) in enumerate(Comods)}
    
    ##Checks and assertions#
    Demands = nx.get_node_attributes(DG2,name='Demand')

    if Y==None:
        assert len(C) == len(MS)
    elif Y!=None:
        assert len(C) == len(DelayArcs)*len(Comods)
    
    #Check all demands are the length of the comodities
    #Demands = list(nx.get_node_attributes(DG2,name='Demand').values())
    assert all([len(Demands[i]) == len(Comods) for i in Demands])
    
    ##Check that ComodGroups and Alpha are the same length as the comodities
    #Assertion 1
    assert len(Comods) == len(Alpha)
    #Assertion 2
    assert len(Comods) == len(ComodGroups)
    
    
    ##Setting up gurobi model
    #Create model object
    m=gp.Model(name="Basic")
    
    

    
    
    #Set up graph and demands for multiple scenarios
    
    #Generate demand in different scenarios
    #Depends on if sources/sinks have been added as related pairs, or added seperately
    #This is indicated with the 'Pairs' argument
    print("Starting Scenario generation")
    if Pairs==True:
        ZDemands=np.round(ZDist(size=[Z,len(Comods)]))
    elif Pairs==False:
        ZDemands, CVs = ZDist(size=[Z,len(Comods)])
        ZDemands=[dict(zip(Comods,z)) for z in ZDemands]
    else:
        assert Pairs in [True,False]
        
    
    ##Add sources and sinks to graph
    
    #Get list of nodes which are sources and sinks
    Sources=[i[0] for i in DG2.nodes(data='WS') if i[1]=='Source']
    Sinks=[i[0] for i in DG2.nodes(data='WS') if i[1]=='Sink']
    
    
    
    
    #Change demand at the source and sink nodes
    print("Changing source/sink demand")
    if Pairs==True:
        for i in Sources:
            Demands[i]=[dict(zip(Comods,j)) for j in ZDemands]
        
        for i in Sinks:
            Demands[i]=[dict(zip(Comods,-j)) for j in ZDemands]
    else:
        for i in Sources:
            Demands[i]=[dict(zip(Comods,[Demands[i][k]*(1+CVs[k]*z[k]) for k in Comods])) for z in ZDemands]
        #In this situation, have to assume there's only one sink, and that meeting deadlines is managed by constraining arc capacities
        #Sinks will be a list of 1, with the sink in it.
        for i in Sinks:
            Demands[i]=[{k:-1*sum([Demands[i][z][k] for i in Sources]) for k in Comods} for z in range(Z)]
        
        
        
    ##Add variables
    print("Adding variables...")
    #Add X variables - X[i,k] is how much of commodity k to send along edge i
    X=m.addVars(DG2.edges,Comods,Z,name="Flow",vtype=VTypes['X'])
    #Set a variable for each edge starting at a manual work area, which controls the capacity of that edge
    
    if Y == None:
        Y=m.addVars(MS,name="Cap",vtype=VTypes['Y'])
        Obj="Workers"
    else:
        Obj="DelayFlow"
    
    print("Adding constraints...")
    ##Add the Mass Balance Constraints
    #Finds the edges leading into and out-of each node, and matches the sum of the flow out- sum of the flow in to the demand for each commodity at each node
    for (n,attr) in DG2.nodes.items():
        if n not in WS:
            print("n is ", n, "Type is ",type(n),"In WS?", n in WS)
        if WS[n] in ['Source','Sink']:
            [[m.addConstr(gp.quicksum(X[u,v,k,z] for (u,v) in DG2.out_edges(n))-gp.quicksum(X[u,v,k,z] for (u,v) in DG2.in_edges(n)) ==Demands[n][z][k], name='MassBal') for k in Comods] for z in range(Z)]
        else:
            for k in Comods:
                for z in range(Z):
                    m.addConstr(gp.quicksum(X[u,v,k,z] for (u,v) in DG2.out_edges(n))-gp.quicksum(X[u,v,k,z] for (u,v) in DG2.in_edges(n)) == Demands[n][k], name='MassBal')
        
    
    ##Arc Capacity Constraints
    #NOTE: X is indexed by edge indices, but TotalCap and ComCap are indexed by edge names - hence the differences in the notation.
    #If the Edge is NOT a manual edge (that is, originating under a manual node), total flow has to be less than the total capacity
                    #Furthermore, total flow of all comodities in the same group (e.g. 1cLetters, etc) across all different sources must be less than the Commodity Capacity for that group
    for (e, e_attrs) in DG2.edges.items():
        u,v=e
        if e_attrs.get('Type') != 'Man':
            [m.addConstr(gp.quicksum(X[u,v,k,z] for k in Comods)<=TotalCap[e]) for z in range(Z)]
            for j in ComodGroups:
                [[m.addConstr(gp.quicksum(X[u,v,k,z] for k in ComodGroups[j])<=ComCap[e][k2]) for z in range(Z)] for k2 in ComodGroups[j]]
        #Otherwise if edge is manual, capacities must be multiplied by the Y variable (number of workers) in the relevant work area (that of the origin of the edge)
        else:
            #Find the workstation of the node
            WhichWS=DG2.nodes[e[0]]['WS']
            #Add the total Capacity Constraint.
            [m.addConstr(gp.quicksum(X[u,v,k,z] for k in Comods)<=TotalCap[e]*Y[WhichWS]) for z in range(Z)]
            #Add the comodity capacity constraints
            for j in ComodGroups:
                [[m.addConstr(gp.quicksum(X[u,v,k,z] for k in ComodGroups[j])<=Y[WhichWS]*ComCap[e][k2]) for z in range(Z)] for k2 in ComodGroups[j]]
    

    
    ##Set objetive function
# =============================================================================
#     if Obj=='Workers':
#         m.setObjective(gp.quicksum(C[i]*Y[i] for i in MS), GRB.MINIMIZE)
#     elif Obj=='DelayFlow':
#         m.setObjective(gp.quicksum(C[i,k]*X[u,v,k,z] for (u,v) in DelayArcs for k in Comods for z in range(Z)), GRB.MINIMIZE)
#     else:
#         assert 1==2
# =============================================================================
    m.setObjective(gp.quicksum(C[i]*Y[i] for i in MS)+float(1/Z)*gp.quicksum(X[e[0],e[1],k,z]*d[e][k] for e in DelayArcs for k in Comods for z in range(Z)))
    
    
    ##Optimise
    if Opt==True:
        print("Beginning optimisation")
        m.setParam("OutputFlag",1)
        m.setParam("Threads",Threads)
        m.optimize()
    
    return X,Y,m,Demands#,AlphaCons
