# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:24:44 2020

@author: thorburh
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:28:56 2020

@author: thorburh
"""

#import itertools
import networkx as nx
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mailopt.data import ProblemData

DG = nx.DiGraph()

DG.add_edges_from([('ImpA','1cImp'),('ImpA','1cManLS'),('ImpA','ManSeg'),('ImpA','2cManLS'),('ImpA','2cImp'),
                   ('1cImp','1cManLS'),
                   ('ManSeg','1cManLS'),('ManSeg','2cManLS'),
                   ('2cImp','2cManLS'),
                   ('1cManLS','Inw'),
                   ('2cManLS','Inw'),
                   ('Inw','Completion')])

# =============================================================================
# plt.plot()
# #<matplotlib.axes._subplots.AxesSubplot object at ...>
# nx.draw(DG, with_labels=True, font_weight='bold')
# plt.close()
# =============================================================================

#%%
##====Try to time-expand this network====

## Define the necessary data
# Comodities
Comods=['1cPost','2cPost']
# Source/sink demands
B=[200 for i in range(len(Comods))]
# Discrete time poinds
Times=[0,1,2,3,4,5,6,7]
# Work areas run by machines
MachineWS=['ImpA', '1cImp', '2cImp']


def TimeExpand(BaseGraph, Times, Comods, B, MachineWS):
    """
    Function to time-expand a graph

    Takes:
    - BaseGraph: A graph (a networkx object) showing the base network of the system
    - Times: A list of discrete time points to expand the network over
    - Comods: A list of comodities (mail types) that have to move across the network
    - B: A list of source/sink demands
    - MachineWS: A list of the work areas (nodes in the base graph) that are mechanical sorting areas. This effects the possible time jumps between these areas
    """
    ## CREATE NEW GRAPH OBJECT
    DG2 = nx.DiGraph()
    
    ## CREATE NODES IN THE NEW GRAPH
    # Create list of Node-Time pairs
    IT2=[(i,j) for i in BaseGraph.nodes() for j in Times]
    # Add these to the created graph
    # Each created node has:
    # - A WS (work station) attribute (str)
    # - A time attribute (int),
    # - A list of ints with the same length as the commodities, giving the demand for each. Initially, these are all set to 0.
    DG2.add_nodes_from([(i, {"WS":IT2[i][0], "Time":IT2[i][1], "Demand":{k:0 for k in Comods}}) for i in range(len(IT2))])
    # Create dictionary of attributes of each node - useful for referencing things later
    #print(DG2.nodes(data='WS'))
    WS=nx.get_node_attributes(DG2,name='WS')
    WSList=list(nx.get_node_attributes(DG2,name='WS').values())
    TimeList=list(nx.get_node_attributes(DG2,name='Time').values())
    NodeAtts=[list(i[1].values()) for i in DG2.nodes(data=True)]
    # Create a dictionary to find a node index using the work station and 
    #PlaceTimeDict={(NodeAtts[i][0],NodeAtts[i][1]):i for i in range(len(NodeAtts))}
    PlaceTimeDict={(WSList[i],TimeList[i]):i for i in range(len(NodeAtts))}
    #print(PlaceTimeDict.keys())
    #print("--------------")
    #print(PlaceTimeDict2.keys())

    
    ## Add the edges
    # First, the easy  - edges between nodes of the same workstation, for increasing time periods
    for i in BaseGraph.nodes():
        #Temp=[j[0] for j in DG2.nodes(data='WS') if j[1]==i]
        Temp=[j for j in DG2.nodes() if i==WS[j]]
        Temp2=[(Temp[i-1],Temp[i]) for i in range(1,len(Temp))]
        DG2.add_edges_from(Temp2)
    
    # nx.draw(DG2, with_labels=True, font_weight='bold')
    
    # MachineWS=['ImpA','1cImp','2cImp']

    # Set edges between different nodes in the base network, and their successors 1 time period later.
    # NB: We set this time gap for ALL nodes. We will control capacity of manual work stations with gurobi variables later.
    for i in list(BaseGraph.nodes()):
        Neighbours=list(BaseGraph.successors(i))
        for k in Neighbours:
            for j in Times[:-1]:
                Node1=PlaceTimeDict[(i,j)]
                Node2=PlaceTimeDict[(k,j+1)]
                DG2.add_edge(Node1,Node2)
            
    
    # nx.draw(DG2, with_labels=True, font_weight='bold')        
    
    ## Add source and sink nodes, with correct demand at each
    
# =============================================================================
#     DG2.add_nodes_from([('Source',{'NetStatus':'Source',"Demand":B})])
#     DG2.add_edge('Source',0)
#     DG2.add_node(PlaceTimeDict['Completion',Times[-1]],WS='Completion',Time=Times[-1],NetStatus='Sink',Demand=[-1*i for i in B])
# =============================================================================
    
    #Class each edge as:
    # - a hold (an edge between a node, and the node corresponding to the same work area 1 time period later),
    # - a Mach (an edge coming from a mechanical work area going to another work area 1 time period later)
    # - a Man (an edge coming from a manual work area, going to another work area at a later time), to classify
    #These are useful for defining cost variables later
    for i in list(DG2.edges()):
        Orig=i[0]
        Dest=i[1]
        #OrigWS,OrigT=NodeAtts[Orig][0:2]
        #DestWS,DestT=NodeAtts[Dest][0:2]
        OrigWS = WSList[Orig]
        DestWS = WSList[Dest]
        if OrigWS==DestWS:
            DG2.add_edges_from([(Orig,Dest,{'Type':'Hold'})])
        elif OrigWS in MachineWS:
            DG2.add_edges_from([(Orig,Dest,{'Type':'Mach'})])
        else:
            DG2.add_edges_from([(Orig,Dest,{'Type':'Man'})])
    
    #Return the graph, dictionary of node attributes, and dictionary of node indices
    return DG2,NodeAtts,PlaceTimeDict

NewGraph=TimeExpand(DG,Times,Comods,B,MachineWS)
NG=NewGraph[0]

def AddSourceSink(Graph, Sources, Sinks, Demands):
    """
    
    Description
    -----------
    Adds desired source and associated sink nodes to a graph, if sources/sinks are in associated pairs with equal and opposite demand
    
    Parameters
    ----------
    Graph : TYPE: Networkx object
        DESCRIPTION: The graph we wish to add sources and sinks to.
    Sources : TYPE: List of tuples
        DESCRIPTION: List of 2-tuples, each giving the name of a source node, and the node it links to.
    Sinks : TYPE: List of tuples
        DESCRIPTION: List of 2-tuples, each giving the name of a sink node, and the node that links to it.
    Demands : TYPE: List of dictionarys
        DESCRIPTION: List of dictionarys. Each dictionary describes the demand associated between each source-sink pair for different commodities (commodities are the keys).

    Returns: 
    -------
    None. Graph is changed after using this function

    """
    #Check all sources, sinks, and Demands have the same length
    assert len(Sources)==len(Demands)
    assert len(Sources)==len(Sinks)
    
    #Add demand of 0 for each commodity for all existing nodes in the graph
    Graph.add_nodes_from([i for i in list(Graph.nodes())],Demand=dict(zip(Demands[0].keys(),[0 for k in Demands[0].values()])))
    #Add a node for each Source in 'Sources', with associated Demand
    Graph.add_nodes_from([(Sources[i][0],{"WS":"Source","Time":nx.get_node_attributes(Graph,name="Time")[Sources[i][1]]-1,"Demand":Demands[i]}) for i in range(len(Sources))])
    #Add a node for each Sink in 'Sinks', with associated negative Demand
    Graph.add_nodes_from([(Sinks[i][0],{"WS":"Sink","Time":nx.get_node_attributes(Graph,name="Time")[Sinks[i][1]]+1,"Demand":dict(zip(Demands[i].keys(),[-1*j for j in Demands[i].values()]))}) for i in range(len(Sources))])
    #Add edges from each source to its associated node in Graph
    Graph.add_edges_from([(Sources[i][0],Sources[i][1]) for i in range(len(Sources))])
    #Add edges to each sink from its associated node in Graph
    Graph.add_edges_from([(Sinks[i][1],Sinks[i][0]) for i in range(len(Sources))])
    

def AddSourceSink2(Graph, Sources, Sinks, Commods, PlaceTimeDict,FinalTime,AddDelays=True):
    """
    
    Description
    -----------
    Adds desired source and sink nodes to a graph, if sources/sinks are are given in two independent lists

    Parameters
    ----------
    Graph : TYPE: Networkx object
        DESCRIPTION: The graph we wish to add sources and sinks to.
    Sources : TYPE List of lists
        DESCRIPTION. Sources and sinks are lists of lists of sources/sinks for each commodity. The entries for each sub-list are:
    (Commodity name, WA number, Time, Amount/demand)
    Sinks : TYPE List of lists
        DESCRIPTION. See 'Sources'
    Commods : TYPE: List of strings
        DESCRIPTION. List of the commodities for the network
    PlaceTimeDict : TYPE: Dictionary
        DESCRIPTION. Dictionary (Output of TimeExpand), indexed by WA numbers and time periods, giving the node associated with that WA number and time period

    Returns
    -------
    None.

    """
    #Source and sink list are lists by commodity/stream. Need to make them into lists by node
    SourceNodes=list(set([PlaceTimeDict[(j[1],j[2])] for i in Sources for j in i]))
    #print('SourceNodes are ',SourceNodes)
    SinkNodes=list(set([PlaceTimeDict[(j[1],j[2])] for i in Sinks for j in i]))
    #print("SinkNodes is ",SinkNodes)
    UWA=set(nx.get_node_attributes(Graph,name='WS').values())
    
    #Add the source nodes
    #This won't work - the first i needs to be a source name instead. Fixed by putting a string in
    #print({k:0 for k in Commods})
    Graph.add_nodes_from([('Source ' + str(i),{"WS":"Source","Time":Graph.nodes(data='Time')[i]-1,"Demand":{k:0 for k in Commods}}) for i in SourceNodes])
    #print(Graph.nodes(data='Demand')[0])
    #Add the sink nodes
    Graph.add_nodes_from([('Sink ' + str(i),{"WS":"Sink","Time":Graph.nodes(data='Time')[i]+1,"Demand":{k:0 for k in Commods}}) for i in SinkNodes])
    #Fix the demand for the source and sink nodes:
    for i in SourceNodes:
        #Find correct streams for this node:
        Streams=[l for j in Sources for l in j if PlaceTimeDict[(l[1],l[2])]==i]
        #print("Source Streams are ",Streams)
        for j in Streams:
            Graph.nodes(data='Demand')['Source ' + str(i)][j[0]]=j[3]
    for i in SinkNodes:
        #Find correct streams for this node:
        Streams=[l for j in Sinks for l in j if PlaceTimeDict[(l[1],l[2])]==i]
        #print("Sink streams are ", Streams)
        for j in Streams:
            Graph.nodes(data='Demand')['Sink ' + str(i)][j[0]]=-j[3]
    ##Check all demands sum to 0
# =============================================================================
#     for k in Commods:
#         assert sum([i[k] for i in Graph.nodes(data='Demand')])==0
# =============================================================================
# =============================================================================
#     for i in Sources:
#         if len(i)>0:
#             for j in i:
#                 SourceNode=list(Graph.predecessors(PlaceTimeDict[(j[1],j[2])]))[0]
#                 print('This source node is ',SourceNode)
#                 #print(i)
#                 #print(j,PlaceTimeDict[(j[1],j[2])],Graph.nodes(data=True)[PlaceTimeDict[(j[1],j[2])]],type(Graph.nodes(data='Demand')[PlaceTimeDict[(j[1],j[2])]]))
#                 Graph.nodes(data='Demand')[SourceNode][j[0]]=j[3]
#     for i in Sinks:
#         if len(i) > 0:
#             for j in i:
#                 SinkNode=list(Graph.sucessors(PlaceTimeDict[(j[1],j[2])]))[0]
#                 Graph.nodes(data='Demand')[SinkNode][j[0]]=-j[3]
# =============================================================================
    #Still need to add the arcs
    Graph.add_edges_from([('Source '+str(i),i) for i in SourceNodes],Type="Source")
    Graph.add_edges_from([(i,'Sink '+str(i)) for i in SinkNodes],Type="Sink")
    #Add delay arcs
    #Get unique work areas
    if AddDelays==True:
        if 'Completion' in UWA:
            UWA.remove('Completion')
        Graph.add_edges_from([(PlaceTimeDict[(j,FinalTime)],'Sink '+str(i)) for i in SinkNodes for j in UWA], Type="Delay")
    
def CondenseNetwork(OldData,NewTime=24):
    """

    Parameters
    ----------
    WarrData : TYPE
        DESCRIPTION.
    NewTime : TYPE, optional
        DESCRIPTION. The default is 24.

    Returns
    -------
    NewData : TYPE
        DESCRIPTION.

    """
    #Check that the current time periods can be divided by the new time periods
    OldTime=len(OldData.Times)
    assert OldTime/NewTime==np.floor(OldTime/NewTime)
    
    #Pull out structures that we're keeping
    NewBaseNetwork=OldData.BaseNetwork
    NewTimes=range(NewTime)
    NewCommods=OldData.Comods
    
    #Find sources and sinks
    
    
    
    
    NewGraph, NodeAtts, PlaceTimeDict = TimeExpand(NewBaseNetwork,NewTimes,NewCommods,[],[])
    TimePlaceDict={PlaceTimeDict[w,t]:(w,t) for (w,t) in PlaceTimeDict}
    
    
    #TEMCNoSourceSink=TEMC.copy()
    ##Adding sources and sinks
    #Get previous ones
    OldSources=OldData.Sources
    OldSinks=OldData.Sinks
    #Copy these into new ones
    NewSources=[[list(ss) for ss in s] for s in OldSources]#copy.deepcopy(OldSources)
    NewSinks=[[list(ss) for ss in s] for s in OldSinks]#copy.deepcopy(OldSinks)
    NewSources2=[]
    NewSinks2=[]
    #Change the times
    for s in NewSources:
        for ss in s:
            ss[2] = np.floor((ss[2])/(OldTime/NewTime))
        KeyList=set([(a,b,c) for (a,b,c,d) in s])
        NewSourceList=[[a,b,c,sum([dd for (aa,bb,cc,dd) in s if (aa,bb,cc)==(a,b,c)])] for (a,b,c) in KeyList]
        NewSources2.append(NewSourceList)
    for s in NewSinks:
        for ss in s:
            ss[2] = np.floor((ss[2])/(OldTime/NewTime))
    
    
    #Add them to the graph
    AddSourceSink2(NewGraph,NewSources2,NewSinks,NewCommods,PlaceTimeDict,NewTimes[-1],AddDelays=True)
    
    NewGraph=AddIDStreamPaths(NewGraph,OldData.IDDicts,OldData.Comods,OldData.StreamPaths,
                              PlaceTimeDict,OldData.WANameNumber,OldData.WANumberName,NewTimes)
    
    #Create the worker caps on the new graph
    OldWorkerCapList=[(w,t,OldData.WorkerCap[w,t]) for (w,t) in OldData.WorkerCap]
    NewWorkerCapList=[(w,np.floor(t/(OldTime/NewTime)),Cap) for (w,t,Cap) in OldWorkerCapList]
    NewWorkerCapSet=set(NewWorkerCapList)
    NewWorkerCapKeySet=set([(k[0],k[1]) for k in NewWorkerCapSet])
    NewWorkerCaps={}
    for (w,t) in NewWorkerCapKeySet:
        AssocCaps=[Cap for (w2,t2,Cap) in NewWorkerCapSet if (w2,t2)==(w,t)]
        NewWorkerCaps[w,t]=max(AssocCaps)
    
    #Create the commodity caps on the new graph
    #Get required mappings and dictionaries
    OldNodeWAs=nx.get_node_attributes(OldData.DG2,'WS')
    OldNodeTimes=nx.get_node_attributes(OldData.DG2,'Time')
    OldPTD={n:(OldNodeWAs[n],OldNodeTimes[n]) for n in OldNodeWAs}
    OldTPD={(OldNodeWAs[n],OldNodeTimes[n]):n for n in OldNodeWAs}
    OldComCapList=[(OldPTD[k[0]][0],OldPTD[k[0]][1],OldPTD[k[1]][0],OldPTD[k[1]][1],list(OldData.ComCap[k].keys()),list(OldData.ComCap[k].values())) for k in OldData.ComCap]
    NewComCapList=[(w1,np.floor(t1/(OldTime/NewTime)),w2,np.floor(t2/(OldTime/NewTime)),ComCapKeys,ComCapVals) for (w1,t1,w2,t2,ComCapKeys,ComCapVals) in OldComCapList]
    NewComCapKeysList=[(w1,t1,w2,t2) for (w1,t1,w2,t2,ComCapKeys,ComCapVals) in NewComCapList]
    NewComCapKeysSources=[(w1,t2-1,w2,t2) for (w1,t1,w2,t2) in NewComCapKeysList if w1=='Source']
    NewComCapKeysSet=set(NewComCapKeysList+NewComCapKeysSources)
    NewComCapKeysSet=[(w1,t1,w2,t2) for (w1,t1,w2,t2) in NewComCapKeysSet if t1!=t2]
    #print(len(NewComCapKeysSet))
    NewComCap={}
    for (w1,t1,w2,t2) in NewComCapKeysSet:
        #print(w1,t1,w2,t2)
        if w1=='Source':
            SmallComCapsKeys=[NewComCapList[0][4]]
            Maxs=[GRB.INFINITY for k in SmallComCapsKeys[0]]
        else:
            SmallComCapsKeys=[ComCapKeys for (w1b,t1b,w2b,t2b,ComCapKeys,ComCapVals) in NewComCapList if (w1b,t1b,w2b,t2b)==(w1,t1,w2,t2)]
            #for KeysList in SmallComCapsKeys:
                #assert KeysList==SmallComCapsKeys[0]
            SmallComCapValues=[ComCapVals for (w1b,t1b,w2b,t2b,ComCapKeys,ComCapVals) in NewComCapList if (w1b,t1b,w2b,t2b)==(w1,t1,w2,t2)]
            SCCVDF=pd.DataFrame(SmallComCapValues)
            Maxs=SCCVDF.max(axis=0).values
        if w1=='Source':
            ConnectedNode=PlaceTimeDict[w2,t2]
            NewKeyOrigin='Source '+str(ConnectedNode)
        else:
            NewKeyOrigin=PlaceTimeDict[w1,t1]
        if w2=='Sink':
            ConnectedNode=PlaceTimeDict['Completion',NewTime-1]
            NewKeyDestination='Sink '+ str(ConnectedNode)
        else:
            NewKeyDestination=PlaceTimeDict[w2,t2]
        #print(SmallComCapsKeys[0])
        #print(Maxs)
# =============================================================================
#         if (NewKeyOrigin,NewKeyDestination)==('Source 1045',1045):
#             print("SUS EDGE")
#             print(dict(zip(SmallComCapsKeys[0],Maxs)))
# =============================================================================
        NewValsDict=dict(zip(SmallComCapsKeys[0],Maxs))
        NewComCap[NewKeyOrigin,NewKeyDestination]=NewValsDict
    
    #Create Node caps for the new graph
    OldNodeCapList=[(OldPTD[n][0],OldPTD[n][1],OldData.NodeCaps[n],n) for n in OldData.NodeCaps]
    NewNodeCapList=[(w,np.floor(t/(OldTime/NewTime)),Cap*(OldTime/NewTime),n) for (w,t,Cap,n) in OldNodeCapList]
    NewNodeCapKeySet=set([(w,t) for (w,t,Cap,n) in NewNodeCapList])
    NewNodeCaps={}
    for (w,t) in NewNodeCapKeySet:
        if w not in ['Source','Sink']:
            AssocCaps=[Cap for (w2,t2,Cap,n) in NewNodeCapList if  (w2,t2)==(w,t)]
            NewNodeCaps[PlaceTimeDict[w,t]]=max(AssocCaps)
        elif w=='Source':
            Nodes=[n for (w2,t2,Cap,n) in NewNodeCapList if  (w2,t2)==(w,t)]
            Successors=[int(n[7:]) for n in Nodes]
            SuccessorPlaceTimes=[(OldNodeWAs[n],OldNodeTimes[n]) for n in Successors]
            NewSuccessorPlaceTimes=[(w2,np.floor(t2/(OldTime/NewTime))) for (w2,t2) in SuccessorPlaceTimes]
            NewSourceNodes=['Source ' + str(PlaceTimeDict[w2,t2]) for (w2,t2) in NewSuccessorPlaceTimes]
            #print(w,t,Nodes)
            #assert all([n==Nodes[0] for n in Nodes])
            for s in NewSourceNodes:
                NewNodeCaps[s]=GRB.INFINITY
        elif w=='Sink':
            Nodes=[n for (w2,t2,Cap,n) in NewNodeCapList if  (w2,t2)==(w,t)]
            Predecessors=[int(s[5:]) for s in Nodes]
            PredPlaceTimes=[(OldNodeWAs[n],OldNodeTimes[n]) for n in Predecessors]
            NewPredPlaceTimes=[(w2,np.floor(t2/(OldTime/NewTime))) for (w2,t2) in PredPlaceTimes]
            NewSinkNodes=['Sink ' + str(PlaceTimeDict[w2,t2]) for (w2,t2) in NewPredPlaceTimes]
            #assert all([n==Nodes[0] for n in Nodes])
            NewNodeCaps[Nodes[0]]=GRB.INFINITY
            
    
        
    
    NewWorkerCapList=[(w,np.floor(t/(OldTime/NewTime)),Cap) for (w,t,Cap) in OldWorkerCapList]
    NewWorkerCapSet=set(NewWorkerCapList)
    NewWorkerCapKeySet=set([(k[0],k[1]) for k in NewWorkerCapSet])
    NewWorkerCaps={}
    for (w,t) in NewWorkerCapKeySet:
        AssocCaps=[Cap for (w2,t2,Cap) in NewWorkerCapSet if (w2,t2)==(w,t)]
        NewWorkerCaps[w,t]=max(AssocCaps)
    
    
    ##Create new shifts
    OldShifts=OldData.Shifts
    ChangeFac=NewTime/OldTime
    NewShifts={s:range(int(OldShifts[s][0]*ChangeFac),int((OldShifts[s][-1]+1)*ChangeFac)) for s in OldShifts}
    
    #NewNodeCaps=[]
    
    NewData=ProblemData(NewGraph,NewTimes,NewComCap,OldData.WorkPlan,NewWorkerCaps,OldData.Comods,
                        OldData.MS,OldData.C,NewNodeCaps,OldData.BaseNetwork,OldData.IDDicts,OldData.ComodGroups,OldData.Tethered,
                        OldData.StreamPaths,OldData.WANameNumber,OldData.WANumberName,OldData.AcceptPaths,OldData.Date,
                        NewSources,NewSinks,NewShifts)
# =============================================================================
#     ProblemData(TEMC,Times,ComCap,WorkPlanProblemData,WorkerCaps,CommodsList,
#                          WAList,Mults,NodeCaps,BaseMC,IDDicts,CommodGroups,TetheredWAs,
#                          StreamPaths,WANameNumber,WANumberName,AcceptPaths,Date,Sources,Sinks)
# =============================================================================
    return NewData


def AddIDStreamPaths(TEMC,IDDicts,CommodsList,StreamPaths,PlaceTimeDict,WANameNumber,WANumberName,Times):
    #Start with adding the direct stream paths
    AcceptPaths={k:[(StreamPaths[k][j],StreamPaths[k][j+1]) for j in range(len(StreamPaths[k])-1)] if k in StreamPaths else [] for k in CommodsList}
    
    ##Now, add the indirect streams
    #1) Make list of all mappings from ID stream origin to destination
    #2) Loop through all commodities - if they pass through (should be end at) an origin of the ID stream, add this to the acceptable paths
    #3) Remove this mapping
    #4) Repeat, until all mappings have been added
    #NB: ONLY WORKS IF THERE ARE NO LOOPS IN THE INDIRECT STREAMS!!!
    
    IDStreamPaths=[(w,d) for w in IDDicts for d in IDDicts[w]['Destinations']]
    Added={p:0 for p in IDStreamPaths}
    while not sum(Added.values())==len(IDStreamPaths):
        print("Rem paths is ",len(IDStreamPaths)-sum(Added.values()))
        #Determine which Current paths stem from currently ending paths
        #PathEnds=[p[1] for k in CommodsList for p in AcceptPaths[k]]
        #NewPaths=[p for p in IDStreamPaths if p[0] in PathEnds]
        for npath in IDStreamPaths:
            for k in AcceptPaths:
                ComodPathEnds=[p[1] for p in AcceptPaths[k]]
                if npath[0] in ComodPathEnds:
                    AcceptPaths[k].append(npath)
                    Added[npath]=1
    
    Score=100
    while Score>0:
        Score=0
        for p in IDStreamPaths:
            for k in AcceptPaths:
                #print(k)
                PathEnds=[t[1] for t in AcceptPaths[k]]
                if p[0] in PathEnds and p not in AcceptPaths[k]:
                    #print("p is ",p,", k is ", k)
                    AcceptPaths[k].append(p)
                    Score+=1
    
    #Finally, add links to the completion work area. Do this by looking at all the acceptable paths, finding which dest WAs aren't starting WAs for a new path, and linking them to 'Completion'
    #Then, reduce it down to the unique ones
    for k in AcceptPaths:
        PathStarts=[p[0] for p in AcceptPaths[k]]
        PathEnds=[p[1] for p in AcceptPaths[k]]
        for p in PathEnds:
            if p not in PathStarts:
                AcceptPaths[k].append((p,'Completion'))
        AcceptPaths[k]=set(AcceptPaths[k])
    
    NewEdges=[]
    for k in AcceptPaths:
        for p in AcceptPaths[k]:
            for t in range(1,len(Times)):
                #assert (PlaceTimeDict[WANameNumber[p[0]],t-1],PlaceTimeDict[WANameNumber[p[1]],t]) in TEMC.edges()
                if (PlaceTimeDict[WANameNumber[p[0]],t-1],PlaceTimeDict[WANameNumber[p[1]],t]) not in TEMC.edges():
                    NewEdges.append((PlaceTimeDict[WANameNumber[p[0]],t-1],PlaceTimeDict[WANameNumber[p[1]],t]))
    TEMC.add_edges_from(NewEdges,Type="Man")
    return TEMC            

if __name__ == '__main__':
    AddSourceSink(NG,[('TestSource',0)],[('TestSink',1)],[dict(zip(['1c','2c'],[1,1]))])

    """
    
    Description
    -----------
    Adds desired source and associated sink nodes to a graph
    
    Parameters
    ----------
    Graph : TYPE: Networkx object
        DESCRIPTION: The graph we wish to add sources and sinks to.
    Sources : TYPE: List of tuples
        DESCRIPTION: List of 2-tuples, each giving the name of a source node, and the node it links to.
    Sinks : TYPE: List of tuples
        DESCRIPTION: List of 2-tuples, each giving the name of a sink node, and the node that links to it.
    Demands : TYPE: List of tuples
        DESCRIPTION: List of n-interger lists, where n is the number of commodities in the network. Each tuple describes the demand associated between each source-sink pair.

    Returns: 
    -------
    None. Graph is changed after using this function

    """




