# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:29:07 2020

@author: thorburh

File to measure KPIs for the flow networks
KPIs measured are:
    - Total fill rate (across all scenarios)
    - Average of the fill rate for each scenario
    - Stock-in Probability
"""

import numpy as np
from Optimising.CCMSGModelOptFunction import *

def CalcKPIs(Flows,Z,DelayArcInds,CompArcInds):
    """
    Function to calculate the total fill rate, average fill rate, and Stock-in probability from flows from a network optimising function

    Parameters
    ----------
    Flows : A dictionary
        A dictionary indexed by arcs in a networkx digraph, and different scenarios
    nScenarios : Integer
        Number of scenarios
    DelayArcInds : List of int
        List of the indexes of the delay arcs for the optimised network
    CompArcInds : List of int
        List of the indexes of the completion arcs of the optimised network

    Returns
    -------
    TotalFillRate : float
        The total fill rate
    AverageFillRate : float
        The average fill rate
    StockInProb : float
        The stock-in probability

    """
    #Calculate the delay flow in each scenario
    TotalDelayFlows=[sum(Flows[i,z] for i in DelayArcInds) for z in range(Z)]
    #Calculate the total flow in each scenario
    TotalFlows=[sum(Flows[i,z] for i in DelayArcInds+CompArcInds) for z in range(Z)]
    #Calculate the total fill rate
    TotalFillRate=1-(sum(TotalDelayFlows)/sum(TotalFlows))
    #Calculate the average fill rate
    AverageFillRate=np.mean([(1-(TotalDelayFlows[z]/TotalFlows[z])) for z in range(Z)])
    #Calculate the stock-in prob
    StockInProb=sum([i == 0 for i in TotalDelayFlows])/Z
    #Return relevant numbers
    return TotalFillRate, AverageFillRate, StockInProb, TotalDelayFlows, TotalFlows
    

##Test
# =============================================================================
# K=0
# FlowTest={}
# FlowTest2={}
# for i in range(len(DG2.edges())):
#     for z in range(nScenarios):
#         FlowTest[i,z]=X[i,K,z].x
#     for z in range(nScenarios2):
#         FlowTest2[i,z]=X2[i,K,z].x
# 
# CompArcInds=[i for i in range(len(DG2.edges())) if list(DG2.edges())[i][1]=='Sink' and WS[list(DG2.edges())[i][0]]=='Completion']
# 
# A=CalcKPIs(FlowTest,nScenarios,DelayArcInds,CompArcInds)
# B=CalcKPIs(FlowTest2,nScenarios2,DelayArcInds,CompArcInds)
# 
# =============================================================================
