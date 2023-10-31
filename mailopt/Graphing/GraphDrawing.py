# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:48:50 2020

@author: thorburh
"""


"""
To try and draw the graph more clearly
"""

#import pygraphviz
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB
# from NetworkBuilding.NetworkBuildingFunction import TimeExpand, AddSourceSink
# from NetworkBuilding import DG, Comods, B, MachineWS
# from DetModelOptFunctionv1 import DetNetworkFlowOpt



def unique(sequence):
    """
    Convenience function for finding a set, without ordering.
    Taken from http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html

    Parameters
    ----------
    sequence : A list
        DESCRIPTION. The list you want the unique values of

    Returns
    -------
    list
        DESCRIPTION. List of unique values (unsorted) from sequence.

    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]



def CreateNodePos(Graph,UniqueWS=None):
    """
    Creates node positions for plotting a time-expanded graph

    Parameters
    ----------
    Graph : TYPE: A networkx graph, where the nodes have attributes WS and Time
        DESCRIPTION. The Time-expanded graph you wish to draw

    Returns
    -------
    Positions : TYPE Dictionary of np.ndarray
        DESCRIPTION. A dictionary of np.ndarrays giving the x and y position of each node

    """
    #Get workstations and times of each need node
    AllWS=list(nx.get_node_attributes(Graph,name='WS').values())
    AllTimes=list(nx.get_node_attributes(Graph,name='Time').values())
    
    #If UniqueWS isn't specified, find it from the AllWS list
    #UniqueWS=['Source', 'ImpA', '1cImp', '1cManLS', 'ManSeg', '2cManLS', '2cImp', 'Inw', 'Completion', 'Sink']
    if UniqueWS==None:
        UniqueWS=unique(AllWS)
    #Set each unique WS to a location
    WSLocations=range(len(UniqueWS))
    
    #Create dictionary for node positions. Values are arrays of x (times) and y (-1 *WS) locations of nodes
    Positions={}
    for i in range(len(list(Graph.nodes()))):
        WSLoc=[-1*j for j in WSLocations if AllWS[i]==UniqueWS[j]]
        print(i,[WSLoc])#,WSLoc[0]])
        Positions[list(Graph.nodes())[i]]=np.array([AllTimes[i],WSLoc[0]])   
    return Positions




def PlotDiGraph(Graph,UniqueWS,FigSize=(20,10),width=1,labels=None,NodeCols='blue'):
    """
    

    Parameters
    ----------
    Graph : TYPE: A networkx graph, where the nodes have attributes WS and Time
        DESCRIPTION. The Time-expanded graph you wish to draw
    UniqueWS : TYPE A list of strings
        DESCRIPTION. A list of all the work stations in the graph, in the order you want them plotted
    FigSize : TYPE, A 2-tuple of integers, optional
        DESCRIPTION. The size of the plotting window. The default is (20,10).
    width : TYPE. Either a float, or a list of integers of length len(Graph.Edges)
        DESCRIPTION. The width of the edges. Defaults as 1
    labels : TYPE. Either None, or a dictionary
        DESCRIPTION. If none, labels default to WA name. Otherwise, needs to be a dictionary, with the nodes of Graph as keys, and the node label as the value

    Returns
    -------
    None.

    """
    #Get Node Positions
    Pos=CreateNodePos(Graph,UniqueWS=UniqueWS)
    #Create figure
    plt.figure(figsize=FigSize)
    #Draw network using node positions
    nx.draw_networkx(Graph,pos=Pos,with_labels=False,ax=None,width=width,node_color=NodeCols)
    #Add labels, which are the work areas
    if labels==None:
        nx.draw_networkx_labels(Graph,pos=Pos,labels=nx.get_node_attributes(Graph,name='WS'))
    else:
        assert len(labels)==len(Graph.nodes())
        nx.draw_networkx_labels(Graph,pos=Pos,labels=labels)
    

    
def CalcEdgeWidths(Flow,Edges,ZTotal,Comod=0,Fun=np.mean,Stnd=False):
    EW={}
    for i in Edges:
        FlowKey=Edges[i]
        EW[i]=Fun([Flow[i[0],i[1],Comod,z] for z in range(ZTotal)])
    if Stnd:
        Max=max(EW.values())
        for i in Edges:
            EW[i]=EW[i]/Max
    return EW


def PlotRandomBaseNetwork(BaseNetwork,WANames,nTiers=3,FigSize=(20,10)):
    
    #Set the number of each in each tier
    #Check we have enough tiers
    nWAs=len(WANames)
    TotalTiers=nTiers+2
    assert TotalTiers<=nWAs
    
    #Work out how many WAs in each tier
    Base, extra = divmod(nWAs-2,nTiers)
    
    #For now, just put all the extra WAs in the ealier tiers
    WAsPerTier=[1]+[Base for i in range(nTiers)]+[1]
    for i in range(nTiers):
        if extra!=0:
            WAsPerTier[i+1]+=1
            extra-=1
        else:
            break
    assert sum(WAsPerTier)==nWAs
    
    WAsInTiers=[[sum(WAsPerTier[:i])+j for j in range(WAsPerTier[i])] for i in range(TotalTiers)]
    
    #Set XCoords
    #All in same tier have the same XCoords
    XCoords=[i for i in range(len(WAsInTiers)) for j in WAsInTiers[i]]
    MaxInTier=np.max(WAsPerTier)
    #Need to work out what the Y
    YCoords=[j/2 for i in WAsInTiers for j in np.arange(start=(len(i)-1)/2,stop=-(len(i)+1)/2,step=-1)]
    
    BothCoords=zip(XCoords,YCoords)
    PositionsDict=dict(zip(list(BaseNetwork.nodes()),BothCoords))
    
    plt.figure()
    nx.draw_networkx(BaseNetwork,pos=PositionsDict,ax=None)


if __name__=='__main__':
    
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

    Times2=[0,1,2,3,4,5,6,7,8,9,10]
    ## Create graph
    DG2, NodeAtts, PlaceTimeDict=TimeExpand(DG, Times2, Comods, B, MachineWS)
    UWS=['Source', 'ImpA', '1cImp', '2cImp', 'ManSeg', '1cManLS',  '2cManLS', 'Inw', 'Completion', 'Sink']
    
    # Draw network without source and sink
    P=CreateNodePos(DG2)
    plt.plot(axis="off")
    nx.draw_networkx_nodes(DG2,pos=P)
    nx.draw_networkx_edges(DG2,alpha=1,pos=P)
    nx.draw_networkx(DG2,pos=P,with_labels=False,ax=None)
    nx.draw_networkx_labels(DG2,pos=P,labels=nx.get_node_attributes(DG2,name='WS'))
    plt.show()

    # =============================================================================
    Sources=[('Source',PlaceTimeDict[('ImpA',0)])]
    Sinks=[('Sink',PlaceTimeDict[('Completion',10)])]
    
    AddSourceSink(DG2,Sources,Sinks,[[1]])

    # Get node positions for plot
    UWS=['Source', 'ImpA', '1cImp', '2cImp', 'ManSeg', '1cManLS',  '2cManLS', 'Inw', 'Completion', 'Sink']
    P2=CreateNodePos(DG2,UniqueWS=UWS)

    # Plot the network
    plt.figure(figsize=(20,10))
    nx.draw_networkx(DG2,pos=P2,with_labels=False,ax=None)
    nx.draw_networkx_labels(DG2,pos=P2,labels=nx.get_node_attributes(DG2,name='WS'))
    plt.show()
    plt.close()

    Sources=[('Source',PlaceTimeDict[('ImpA',0)]) for i in range(len(DG.nodes()))]
    Sinks=[('Sink',PlaceTimeDict[(i,10)]) for i in list(DG.nodes())]

    AddSourceSink(DG2,Sources,Sinks,[[1] for i in range(len(DG.nodes()))])
    # Get node positions for plot
    UWS=['Source', 'ImpA', '1cImp', '2cImp', 'ManSeg', '1cManLS',  '2cManLS', 'Inw', 'Completion', 'Sink']
    P2=CreateNodePos(DG2,UniqueWS=UWS)
    P2['Sink']=np.array([11,-8])
    
    # Plot the network
    plt.figure(figsize=(20,10))
    nx.draw_networkx(DG2,pos=P2,with_labels=False,ax=None)
    nx.draw_networkx_labels(DG2,pos=P2,labels=nx.get_node_attributes(DG2,name='WS'))
    plt.show()
    plt.close()
    
    # =============================================================================
    # =============================================================================
    Sources=[('Source',PlaceTimeDict[('ImpA',0)]),
             ('Source',PlaceTimeDict[('ImpA',0)]),
             ('Source',PlaceTimeDict[('ImpA',0)])]

    Sinks=[('Sink1',PlaceTimeDict[('Completion',6)]),
           ('Sink2',PlaceTimeDict[('Completion',8)]),
           ('Sink3',PlaceTimeDict[('Completion',10)])]
    # =============================================================================
