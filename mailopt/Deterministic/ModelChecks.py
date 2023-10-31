# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:39:23 2022

@author: thorburh
"""

import pandas as pd
from os import path
import numpy as np
import networkx as nx
import pkg_resources
import datetime

##File to check a given solution to the optimisation model

##Check mass balance constraints


def CheckModel(WarrModel):
    WAsFilePath=pkg_resources.resource_filename('mailopt','Data/WAs_Hamish.csv')
    WAs=pd.read_csv(WAsFilePath)#pd.read_csv(path.join(FilePath,"WAs_Hamish.csv"))
    WANumberName=dict(zip(WAs['Work Area Number'],WAs['Work Area Name']))
    WANumberName['Source']='Source'
    WANumberName['Sink']='Sink'
    WANameNumber=dict(zip(WAs['Work Area Name'],WAs['Work Area Number']))
    WANameNumber['Completion']='Completion'
    WANumberName['Completion']='Completion'
    NodeWAs=nx.get_node_attributes(WarrModel.Data.DG2,'WS')
    NodeTimes=nx.get_node_attributes(WarrModel.Data.DG2,'Time')
    EdgeTypes=nx.get_edge_attributes(WarrModel.Data.DG2,'Type')
    PlaceTimeDict={(NodeWAs[n],NodeTimes[n]):n for n in NodeWAs}
    NodeDicts={n:(NodeWAs[n],NodeTimes[n]) for n in NodeWAs}
    Date=WarrModel.Data.Date
    DateList=Date.split('/')
    DayNum=datetime.datetime(int(DateList[2]),int(DateList[1]),int(DateList[0])).weekday()+1
    
    #Check that the flows are all going along acceptable paths
    for (u,v,k) in WarrModel.X:
        if WarrModel.X[u,v,k].x>0:
            OriginWA=WANumberName[NodeWAs[u]]
            DestWA=WANumberName[NodeWAs[v]]
            #Case 1 - Origin and destination are the same:
            if OriginWA==DestWA:
                assert NodeTimes[u]==NodeTimes[v]-1
            elif OriginWA=='Source':
                1
            elif DestWA=='Sink':
                if OriginWA!='Completion':
                    assert NodeTimes[u]==143
                    assert EdgeTypes[(u,v)]=='Delay'
            else:
                if (OriginWA,DestWA) not in WarrModel.Data.AcceptPaths[k]:
                    print(u,v,k)
                    assert (OriginWA,DestWA) in WarrModel.Data.AcceptPaths[k]
    print("Flows all along acceptable paths")
    
    ##Check mass balance
    for n in WarrModel.Data.DG2.nodes():
        InEdges=WarrModel.Data.DG2.in_edges(n)
        OutEdges=WarrModel.Data.DG2.in_edges(n)
        for k in WarrModel.Data.Comods:
            FlowsIn=sum([WarrModel.X[e[0],e[1],k].x for e in InEdges if (e[0],e[1],k) in WarrModel.X])
            FlowsOut=sum([WarrModel.X[e[0],e[1],k].x for e in OutEdges if (e[0],e[1],k) in WarrModel.X])
            assert FlowsIn==FlowsOut
    print("All mass balance met")
    tol=0.0001
    #Check ID stream flows:
    for w in WarrModel.Data.IDDicts:
        WANumber=WANameNumber[w]
        InFlowEdges=[(u,v) for (u,v,Type) in WarrModel.Data.DG2.edges(data='Type') if NodeWAs[v]==WANumber and Type!='Hold']
        assert len(InFlowEdges)>0
        DelayEdge=(PlaceTimeDict[WANumber,143],'Sink 7775')
        for k in WarrModel.Data.Comods:
            for d in range(len(WarrModel.Data.IDDicts[w]['Destinations'])):
                DestWANumber=WANameNumber[WarrModel.Data.IDDicts[w]['Destinations'][d]]
                DestRatio=WarrModel.Data.IDDicts[w]['Ratios'][d]
                OutFlowEdges=[(u,v) for (u,v,Type) in WarrModel.Data.DG2.edges(data='Type') if NodeWAs[u]==WANumber and NodeWAs[v]==DestWANumber]
                assert len(OutFlowEdges)>0
                FlowIn=sum([WarrModel.X[e[0],e[1],k].x for e in InFlowEdges if (e[0],e[1],k) in WarrModel.X])
                if (DelayEdge[0],DelayEdge[1],k) in WarrModel.X:
                    DelayFlowOut=WarrModel.X[DelayEdge[0],DelayEdge[1],k].x
                else:
                    DelayFlowOut=0
                FlowOut=sum([WarrModel.X[e[0],e[1],k].x for e in OutFlowEdges if (e[0],e[1],k) in WarrModel.X])
                assert np.abs((FlowIn-DelayFlowOut)*DestRatio-FlowOut)<=tol
    print("ID streams split appropriately")            
    #Check Starts and Finishes are being properly set up
    M=10e6
    #Check tethering first
    FirstWAs=[i[1] for i in WarrModel.Data.Tethered]
    SecondWAs=[i[0] for i in WarrModel.Data.Tethered]
    for i in range(len(WarrModel.Data.Tethered)):
        for t in range(144):
            if WarrModel.Y[FirstWAs[i],t].x>0:
                assert WarrModel.Y[SecondWAs[i],t].x==0
            elif WarrModel.Y[SecondWAs[i],t].x>0:
                assert WarrModel.Y[FirstWAs[i],t].x==0
    print("Tethered WAs are starting/finishing appropriately")
    #Now check earliest starts and latest finishes:
    FilePath=path.join(path.dirname(path.dirname(__file__)),'Data')
    DSConstraints=pd.read_csv(path.join(FilePath,'DS_Constraints_Hamish.csv'))
    StartsDF=pd.read_csv(path.join(FilePath,'Earliest_Starts_Hamish.csv'))
    FinishesDF=pd.read_csv(path.join(FilePath,'Latest_Finishes_Hamish.csv'))
    T1s={'Early':0,'Late':48,'Night':96}
    T2s={'Early':48,'Late':96,'Night':144}
    for w in StartsDF['Work Area Name']:
        #Don't do 'Inward Tracked' - I can't remember why, but I've removed this from the ds constraints
        if w!='Inward Tracked':  
            for Shift in ['Early','Late','Night']:
                ColumnName='Day '+str(DayNum)+' '+Shift
                StartCO=StartsDF[ColumnName][StartsDF['Work Area Name']==w].values[0]
                if type(StartCO)==type('a'):
                    if ":" in StartCO:
                        Hours=int(StartCO.split(":")[0])
                        Minutes=int(StartCO.split(":")[1])
                        TimeInd=0
                        if Hours>=6:
                            TimeInd+=(Hours-6)*6
                        else:
                            TimeInd+=(18+Hours)*6
                        TimeInd+=(Minutes/10)
                        #print("Wall time start is ",StartCO,", TimeInd is ",TimeInd)
                        assert TimeInd==np.floor(TimeInd)
                        #print("WA is ",w,", Shift is ",Shift)
                        if sum(WarrModel.Y[WANameNumber[w],t].x for t in range(T1s[Shift],int(TimeInd)))!=0:
                            print("Sum is ",sum(WarrModel.Y[WANameNumber[w],t].x for t in range(T1s[Shift],int(TimeInd))))
                            assert sum(WarrModel.Y[WANameNumber[w],t].x for t in range(T1s[Shift],int(TimeInd)))==0
                FinishCO=FinishesDF[ColumnName][StartsDF['Work Area Name']==w].values[0]
                if type(FinishCO)==type('a'):
                    if ":" in FinishCO and FinishCO!="00:00":
                        Hours=int(FinishCO.split(":")[0])
                        Minutes=int(FinishCO.split(":")[1])
                        TimeInd=0
                        if Hours==6 and TimeInd==0:
                            TimeInd=144
                        if Hours>=6:
                            TimeInd+=(Hours-6)*6
                        else:
                            TimeInd+=(18+Hours)*6
                        TimeInd+=(Minutes/10)
                        #print("Wall time finish is ",FinishCO,", TimeInd is ",TimeInd)
                        assert TimeInd==np.floor(TimeInd)
                        assert sum(WarrModel.Y[WANameNumber[w],t].x for t in range(int(TimeInd),T2s[Shift]))==0
    print("Earliest start, latest finish happening")
    
    #Check the shift Targets
    #Now, check the H, A, and Z variables are doing what they're supposed to
    #H and A first
    ShiftNames=['Early','Late','Night']
    ShiftTimes=dict(zip(ShiftNames,[47,95,143]))
    tol=0.001
    for Shift in ShiftNames:
        t=ShiftTimes[Shift]
        for w in WarrModel.Data.WorkPlan['WA_Number'].values:
            if (w,t) in PlaceTimeDict:#IS THIS A DANGEROUS IF STATEMENT?
                Target=WarrModel.Data.WorkPlan[WarrModel.Data.WorkPlan.WA_Number==w][Shift].values[0]
                ShiftStart=t-48+1
                ShiftNodes=[PlaceTimeDict[w,tt] for tt in range(ShiftStart,t+1)]
                assert len(ShiftNodes)==48
                if Shift=='Night':
                    OutFlowEdge=(PlaceTimeDict[w,t],'Sink 7775')
                else:
                    OutFlowEdge=(PlaceTimeDict[w,t],PlaceTimeDict[w,t]+1)
                (uu,vv)=OutFlowEdge
                OutFlow=sum(WarrModel.X[uu,vv,k].x for k in WarrModel.Data.Comods if (uu,vv,k) in WarrModel.X)
                if Target=='All':
                    assert OutFlowEdge in WarrModel.PenArcs
                    if OutFlow>0:
                        if not all([np.abs(WarrModel.Y[w,tt].x-WarrModel.Y[w,tt].ub)<tol for tt in range(ShiftStart,t+1)]):
                            print("w is ",w,", t is ",t,", Target is ",Target,", WARNING:NOT USING ALL WORKERS")
# =============================================================================
#                             for tt in range(ShiftStart,t+1):
#                                 print(WarrModel.Y[w,tt].x,WarrModel.Y[w,tt].ub)
# =============================================================================
                elif Target==0:
                    if not all([WarrModel.Y[w,tt].x<tol for tt in range(ShiftStart,t+1)]):
                        print("w is ",w,", t is ",t,", Target is ",Target,", SOME WORKERS BEING SET WHEN THEY SHOULDN'T")
                        print([(tt,WarrModel.Y[w,tt].x,WarrModel.Y[w,tt].ub,tol) for tt in range(ShiftStart,t+1)])
                        assert 1==2
                if Target not in [0,'All']:
                    print("Partial Target")
                    InEdges=[(u,v) for n in ShiftNodes for (u,v) in WarrModel.Data.DG2.in_edges(n) if NodeWAs[u]!=NodeWAs[v]]
                    if t!=47:
                        InEdges+=[(PlaceTimeDict[w,ShiftStart]-1,PlaceTimeDict[w,ShiftStart])]
                    FlowIn=sum([WarrModel.X[u,v,k].x for k in WarrModel.Data.Comods for (u,v) in InEdges if (u,v,k) in WarrModel.X])
# =============================================================================
#                     if FlowIn>Target and WarrModel.H[w,t].x>0.99:
#                         print("w is ",w,", t is ",t,", FlowIn is ",FlowIn,", Target is ",Target,", H is ",WarrModel.H[w,t].x)
#                         assert 1==2
#                     elif FlowIn<=Target and WarrModel.H[w,t].x<0.01:
#                         print("w is ",w,", t is ",t,", FlowIn is ",FlowIn,", Target is ",Target,", H is ",WarrModel.H[w,t].x)
#                         assert 1==2
#                     elif WarrModel.H[w,t].x>0.01 and WarrModel.H[w,t].x<0.99:
#                         print("w is ",w,", t is ",t,", H is ",WarrModel.H[w,t].x,", NOT BINARY")
#                         assert 1==2
#                     #Now, check As
#                     if np.abs(FlowIn-WarrModel.A[w,t].x)>tol and Target > FlowIn:
#                         print("w is ",w,", t is ",t,", FlowIn is ",FlowIn,", A is ",WarrModel.A[w,t].x,", Target is ",Target,", H is ",WarrModel.H[w,t].x)
#                         assert np.abs(FlowIn-WarrModel.A[w,t].x)<=tol
# =============================================================================
                    #Finally, check Zs
                    if Shift=='Night':
                        OutFlowEdge=(PlaceTimeDict[w,t],'Sink 7775')
                    else:
                        OutFlowEdge=(PlaceTimeDict[w,t],PlaceTimeDict[w,t]+1)
                    (uu,vv)=OutFlowEdge
                    OutFlow=sum(WarrModel.X[uu,vv,k].x for k in WarrModel.Data.Comods if (uu,vv,k) in WarrModel.X)
                    OutEdges=[(u,v) for n in ShiftNodes for (u,v) in WarrModel.Data.DG2.out_edges(n) if NodeWAs[u]!=NodeWAs[v] and NodeWAs[v]!='Sink']
                    FlowsOut=sum([WarrModel.X[u,v,k].x for k in WarrModel.Data.Comods for (u,v) in OutEdges if (u,v,k) in WarrModel.X])
                    print("w is ",w,", t is ",t,", FlowIn is ",FlowIn,", Target is ",Target,", FlowsOut is ",FlowsOut,", DelayedMail is ",OutFlow,", Z is",WarrModel.Z[w,t].x)
                
                
                
