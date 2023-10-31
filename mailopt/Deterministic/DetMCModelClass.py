# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:26:05 2021

@author: thorburh

File to create a class for a Deterministic Mail Centre model
"""

from mailopt.data import ProblemData
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mailopt.NetworkBuilding.CreateSmallInstance import CreateSmallNetworkInstance
from mailopt.Deterministic.ModelChecks import CheckModel
import networkx as nx
import itertools

class DetMCModel:
    """
    Class to represent a deterministic mail centre model in gurobi
    
    Attributes:
        Model(gurobipy.Model): A gurobi model object to optimise
        Data (mailopt.ProblemData): A ProblemData class giving the data for the instance we want to optimise
        VTypes (Dict[str]): Dictionary giving variable types. Must have keys 'X' and 'Y', values can be one of 'C' (continuous), 'I' (integer), or 'B' (binary)
        X (gurobipy.Var): The flow variables for the model
        Y (gurobipy.Var): The staff level variables for the model
        G (gurobipy.Var): The max levels for the model (default set to 0 - only used for some objective functions)
        S (gurobipy.Var): The binary variables indicating the start time of a work area
        F (gurobipy.Var): The binary variables indicating the finish time of a work area
    """
    #Create the attributes
    def __init__(self, CentreData,VTypes={"X":'C',"Y":'C'},DelayFlowPen=100):
        print("Start to build the model")
        self.Data=CentreData
        print("1")
        self.Model=gp.Model()
        print("2")
        if self.Data.ComodGroups is None:
            self.Data.ComodGroups = {i: [k] for (i, k) in enumerate(self.Data.Comods)}
        ##Checks and assertions
        assert len(self.Data.C) == len(self.Data.MS)
        
        self.VTypes=VTypes
        
        self.PenArcs=[]
        
        self.Z=gp.tupledict()
        
        self.H=gp.tupledict()
        
        self.A=gp.tupledict()
        
        self.DelayFlowPen=DelayFlowPen
        
        WS=nx.get_node_attributes(self.Data.DG2,name='WS')
        NodeTimes=nx.get_node_attributes(self.Data.DG2,name='Time')
        print("Times are ",set(NodeTimes.values()))
        PlaceTimeDict={(WS[n],NodeTimes[n]):n for n in WS}
        PlaceTimeDict2={n:(WS[n],NodeTimes[n]) for n in WS}
        SourceNodes=[n for (n,atts) in self.Data.DG2.nodes.items() if atts['WS']=='Source']
        Demands=nx.get_node_attributes(self.Data.DG2,'Demand')
        ComArrivals={k:sum(Demands[n][k] for n in SourceNodes) for k in self.Data.Comods}
        
        print("Add X variables")
        # Add X variables - X[u,v,k] is how much of commodity k to send along edge (u,v)
        self.X={}
        for (u,v) in self.Data.DG2.edges:
            for k in self.Data.Comods:
                if ComArrivals[k]>0:
                    if WS[u]==WS[v]:
                        VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)
                        self.X[u,v,k]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                    elif WS[v]=='Sink':
                        VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)
                        self.X[u,v,k]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                    else:
                        if self.Data.ComCap[(u,v)][k]==0:
                            pass#self.X[u,v,k]=0
                        elif self.Data.NodeCaps[u]==0:
                            pass#self.X[u,v,k]=0
                        elif PlaceTimeDict2[u] in self.Data.WorkerCap and self.Data.WorkerCap[PlaceTimeDict2[u]]==0:
                            pass#self.X[u,v,k]=0
                        elif self.Data.ComCap[(u,v)][k]>0:
                            VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)
                            self.X[u,v,k]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                        else:
                            print("Missing X val category for ",(u,v,k))
                            assert 1==2
                else:
                    pass#self.X[u,v,k]=0
        #self.X = self.Model.addVars(self.Data.DG2.edges, self.Data.Comods, name='Flow', vtype=self.VTypes['X'])

        print("Add Y variables")
        # Set a variable for each edge starting at a manual work area, which controls the capacity of that edge
        self.Y={}
        for w in self.Data.MS:
            for t in self.Data.Times:
                if (w,t) not in PlaceTimeDict:
                    self.Y[w,t]=0
                else:
                    n=PlaceTimeDict[w,t]
                    if self.Data.WorkerCap[w,t]==0:
                        self.Y[w,t]=0
                    elif self.Data.NodeCaps[n]==0:
                        self.Y[w,t]=0
                    else:
                        VarName="Cap_"+str(w)+"_"+str(t)
                        self.Y[w,t] = self.Model.addVar(name=VarName, vtype=self.VTypes['Y'],ub=self.Data.WorkerCap[w,t])
        
        # Set variable to control changes in shifts (NOT USED FOR EVERY OBJECTIVE)
        #Set a G variable for controlling smoothness.
        self.G = None
        
        #Update the model
        self.Model.update()
        #Create set of precendent and anticedent work areas
        PrecWAs=[i[1] for i in self.Data.Tethered]
        AntiWAs=[i[0] for i in self.Data.Tethered]
    
        #Create Start and Finish binary variables
        self.S=self.Model.addVars(AntiWAs,self.Data.Times,name="Starts",vtype="B")
        self.F=self.Model.addVars(PrecWAs,self.Data.Times,name="Finishes",vtype="B")
        
        print("Add Mass Balance Constraints")
        ## Add the Mass Balance Constraints
        # Finds the edges leading into and out-of each node, and matches the sum of the flow out- sum of the flow in to the demand for each commodity at each node
        for (n,attrs) in self.Data.DG2.nodes.items():
            self.Model.addConstrs((gp.quicksum(self.X[u,v,k] for (u,v) in self.Data.DG2.out_edges(n) if (u,v,k) in self.X) - gp.quicksum(self.X[u,v,k] for (u,v) in self.Data.DG2.in_edges(n) if (u,v,k) in self.X) == attrs['Demand'][k]
                          for  k in self.Data.Comods if ComArrivals[k]>0), name='MassBal_'+str(attrs['WS'])+"_"+str(attrs['Time']))
        
        print("Add Arc Capacity Constraints")
        ## Arc Capacity Constraints
        for (e, e_attrs) in self.Data.DG2.edges.items():
            u,v=e
            #If any flow is allowed on the arc, add the commodity capacities
            if self.Data.NodeCaps[u]>0:
                ComList4Edge=[k for k in self.Data.Comods if (u,v,k)  in self.X]
                self.Model.addConstrs((self.X[u,v,k] <= self.Data.ComCap[u,v][k] for k in ComList4Edge),name="ComCap")
        ## Add total workflow capacity constraints
        # That is, ALL edges coming from a node (which aren't the hold edge) need to be constrained by the throughput
        for (n,attrs) in self.Data.DG2.nodes.items():
            ExitEdges=[(u,v) for (u,v,Type) in self.Data.DG2.out_edges(n,data='Type') if Type not in ["Hold","Delay"]]
            w=WS[n]
            t=NodeTimes[n]
            if attrs['WS'] not in self.Data.MS+['Source','Sink']:
                LHS=gp.quicksum(self.X[u,v,k] for (u,v) in ExitEdges for k in self.Data.Comods if (u,v,k) in self.X)
                if LHS.size()>0:
                    self.Model.addConstr(LHS<=self.Data.NodeCaps[n],name="NodeCap")
            elif attrs['WS'] in self.Data.MS:
                WhichWS=self.Data.DG2.nodes[n]['WS']
                WhichTime=self.Data.DG2.nodes[n]['Time']
                LHS=gp.quicksum(self.X[u,v,k] for (u,v) in ExitEdges for k in self.Data.Comods if (u,v,k) in self.X)
                if LHS.size()>0:
                    self.Model.addConstr(LHS<=self.Y[WhichWS,WhichTime]*self.Data.NodeCaps[n],name="NodeCap_"+str(n))
            else:
                if WS[n] not in ['Source','Sink']:
                    print("N is ",n)
                    assert 1==2
        print("Add ID stream constraints")
        #Add indirect stream constraints
        for i in self.Data.IDDicts:
            InFlows=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[v]==self.Data.WANameNumber[i] and Type!='Hold']
            #print("i is ",i,", len(InFlows) is ",len(InFlows))
            InFlowSums={k:gp.quicksum(self.X[u,v,k] for (u,v) in InFlows if (u,v,k) in self.X) for k in self.Data.Comods}
            DelayEdges=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[u]==self.Data.WANameNumber[i] and Type=='Delay']
            #print(DelayEdges)
            assert len(DelayEdges)==1
            DelayEdge=DelayEdges[0]
            for d in range(len(self.Data.IDDicts[i]['Destinations'])):
                DestWA=self.Data.IDDicts[i]['Destinations'][d]
                DestRatio=self.Data.IDDicts[i]['Ratios'][d]
                OutFlows=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[u]==self.Data.WANameNumber[i] and WS[v]==self.Data.WANameNumber[DestWA]]
                #print("i is ",i,", len(OutFlows) is ",len(OutFlows), ", DestWA is ", DestWA,', DestRatio is ',DestRatio)
                OutFlowSums={k:gp.quicksum(self.X[u,v,k] for (u,v) in OutFlows if (u,v,k) in self.X) for k in self.Data.Comods}
                for k in self.Data.Comods:
                    if (DelayEdge[0],DelayEdge[1],k) in self.X:
                        #print("Flow Exists",(DelayEdge[0],DelayEdge[1],k))
                        self.Model.addConstr(OutFlowSums[k]==DestRatio*(InFlowSums[k]-self.X[DelayEdge[0],DelayEdge[1],k]),name='ID'+str(i)+str(DestWA)+str(k))
                    else:
                        #print("Flow Doesn't Exist",(DelayEdge[0],DelayEdge[1],k))
                        self.Model.addConstr(OutFlowSums[k]==DestRatio*(InFlowSums[k]),name='ID'+str(i)+str(DestWA)+str(k))
                #[self.Model.addConstr(OutFlowSums[k]==DestRatio*(InFlowSums[k]-self.X[DelayEdge[0],DelayEdge[1],k]),name='ID'+str(i)+str(DestWA)+str(k)) for k in self.Data.Comods]
        #Constrain the g's to be more than the number of workers
        #[[self.Model.addConstr(G[w] >= self.Y[w,t]) for t in self.Data.Times] for w in self.Data.MS]
        print("Add tethering Constraints")
        ##Tethering constraints
        #Continuous starts and finishes
        self.Model.addConstrs((self.S[w,t+1] >= self.S[w,t] for w in AntiWAs for t in self.Data.Times[:-1]),name="StartTimes")
        self.Model.addConstrs((self.F[w,t+1] >= self.F[w,t] for w in PrecWAs for t in self.Data.Times[:-1]),name="FinishTimes")
        
        #Constrainting workers in started/finished WAs
        M=1000#000
        self.Model.addConstrs((self.Y[w,t]<=M*self.S[w,t] for w in AntiWAs for t in self.Data.Times),name="WorkerStarts")
        self.Model.addConstrs((self.Y[w,t]<=M*(1-self.F[w,t]) for w in PrecWAs for t in self.Data.Times),name="WorkerFinishes")
        
        #Enforcing the tethering
        self.Model.addConstrs((self.F[w[1],t] >= self.S[w[0],t+1] for w in self.Data.Tethered for t in self.Data.Times[:-1]),name="Tethering")
        
        #Deal with the shift priorities
        #First, create dictionary mapping shifts to end times
        TimesLen=len(self.Data.Times)
        ShiftLen=int(TimesLen/3)
        EarlyShiftEnd=ShiftLen -1
        LateShiftEnd=(2*ShiftLen) -1
        NightShiftEnd=3*ShiftLen - 1
        ShiftEnds={'Early':EarlyShiftEnd,
                   'Late':LateShiftEnd,
                   'Night':NightShiftEnd}
        #Now, loop through the WorkPlan WA Numbers and Shifts
        for w in self.Data.WorkPlan['WA_Number'].values:
            for Shift in ['Early','Late','Night']:
                Priority=self.Data.WorkPlan[self.Data.WorkPlan.WA_Number==w][Shift].values[0]
                t=ShiftEnds[Shift]
                #If Priority is 0, just assert that upper bound on Y is 0
                if Priority==0:
                    ShiftStartTime=t-ShiftLen+1
                    for tt in range(ShiftStartTime,t+1):
                        assert self.Y[w,tt]==0       
                #If Priority is all, find the edge for the hold/delay arc for this shift, and add it to the PenArcs List
                elif Priority=='All':
                    if w in set(WS.values()):
                        Origin=PlaceTimeDict[w,t]
                        if t==NightShiftEnd:
                            Dest=PlaceTimeDict['Sink',NightShiftEnd+1]
                        else:
                            Dest=PlaceTimeDict[w,t+1]
                        e=(Origin,Dest)
                        self.PenArcs.append(e)
                #If there's an arbitrary target, things get complicated
                else:
                    #Firstly, assert that the priority is a number, not a string
                    assert type(Priority)==type(5)
                    #Now (for ease of readability), rename Priority as 'Target'
                    Target=Priority
                    #Create a Z variable - this is the amount of flow we 'missed' by. This flow needs to be punished in the objective
                    self.Z[w,t]=self.Model.addVar(vtype='C',name='Z_'+str(w)+'_'+str(t),lb=0)
                    #Create an H variable - binary, to indicate if the flow into the WA is less than the target
                    #self.H[w,t]=self.Model.addVar(vtype='B',name='H_'+str(w)+'_'+str(t))
                    #Create an A variable - continous, gives the value of the Flow in (if less than Target), or 0 otherwise
                    #self.A[w,t]=self.Model.addVar(vtype='C',name='A_'+str(w)+'_'+str(t),lb=0)
                    #Calculate the time the shift starts
                    ShiftStart=t-ShiftLen+1
                    #Get a list of all edges (which aren't holding edges) which send flow into the WA
                    EntryEdges=[(u,v) for tt in range(ShiftStart,t+1) for (u,v) in self.Data.DG2.in_edges(PlaceTimeDict[w,tt]) if WS[u]!=WS[v]]
                    #If it's not the first shift, also include the holding edge from the previous shift (for the leftover mail)
                    if t!= EarlyShiftEnd:
                        EntryEdges+=[(PlaceTimeDict[w,ShiftStart-1],PlaceTimeDict[w,ShiftStart])]
                    #Get a list of all edges (which aren't holding edges) which send flow out of the WA (but not to the sink - only relevant for the final shift)
                    ExitEdges=[(u,v) for tt in range(ShiftStart,t+1) for (u,v) in self.Data.DG2.out_edges(PlaceTimeDict[w,tt]) if WS[u]!=WS[v] and WS[v]!='Sink']
                    #Using these edges, calculate the flows into and out of the work area
                    FlowIn=gp.quicksum(self.X[u,v,k] for (u,v) in EntryEdges for k in self.Data.Comods if (u,v,k) in self.X)
                    FlowOut=gp.quicksum(self.X[u,v,k] for (u,v) in ExitEdges for k in self.Data.Comods if (u,v,k) in self.X)
                    #Set a large arbitrary constant
                    M2=1000000000
                    #Set the H constraints
# =============================================================================
#                     self.Model.addConstr(M2*(1-self.H[w,t])>=FlowIn-Target,name='HCons_'+str(w)+'_'+str(t)+'A')
#                     self.Model.addConstr(M2*self.H[w,t]>=Target-FlowIn,name='HCons_'+str(w)+'_'+str(t)+'B')
#                     #Set the A constraints
#                     self.Model.addConstr(self.A[w,t]<=M2*self.H[w,t])
#                     self.Model.addConstr(self.A[w,t]<=FlowIn)
#                     self.Model.addConstr(self.A[w,t]+M2*(1-self.H[w,t]) >= FlowIn)
# =============================================================================
                    #Set the Z constraints
                    self.Model.addConstr(self.Z[w,t]+FlowOut>=Target,name="ZCons_"+str(w)+"_"+str(t))
                    #self.Model.addConstr(self.Z[w,t]+FlowOut>=self.A[w,t])
                
        
        #Update the Model
        self.Model.update()
    
    AcceptableObjs=['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
        
    def create_objective(self,Obj):
        #Check Obj is one of the acceptable arguments
        assert Obj in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
        
        #Optimising for minimum worker cost
        if Obj=='MinCost':
            #Set the Objective
            ObjExp=gp.quicksum(self.Data.C[w]*self.Y[w,t] for w in self.Data.MS for t in self.Data.Times)
        
        #Set objective for minimising the maximum number of workers
        elif Obj=='MinMaxWorker':
            self.G=self.Model.addVars(self.Data.MS, name='GVar_MinMax',vtype='C')
            [[self.Model.addConstr(self.G[w] >= self.Y[w,t],name="GCons") for t in self.Data.Times] for w in self.Data.MS]
            ObjExp = gp.quicksum(self.G[w] for w in self.Data.MS)
        #Set objective for minimising total number of workers in the mail centre at any one time
        elif Obj=='MinMaxWorkerTime':
            self.G=self.Model.addVar(name='GVar_MinMaxTime',vtype='C')
            [self.Model.addConstr(self.G >= gp.quicksum(self.Data.C[w]*self.Y[w,t] for w in self.Data.MS),name="GCons_MinMaxTime_"+str(t)) for t in self.Data.Times]
            ObjExp = self.G
        #Set objective for minimising total number of workers in the mail centre at any one time, BY SHIFT    
        elif Obj=='MinMaxWorkerTimeShift':
            self.G = self.Model.addVars(self.Data.Shifts,name='GVar_MinMaxTime',vtype='C')
            [[self.Model.addConstr(self.G[s] >= gp.quicksum(self.Data.C[w]*self.Y[w,t] for w in self.Data.MS),name="GCons_MinMaxTime_"+str(t)) for t in self.Data.Shifts[s]] for s in self.Data.Shifts]
            ObjExp = gp.quicksum(self.G)
        #Set the objective for minimising the total change in workers
        elif Obj=='MinChange':
            self.G=self.Model.addVars(self.Data.MS,self.Data.Times[1:], name='GVar_MinChange',vtype='C')
            [[self.Model.addConstr(self.G[w,t] >= (self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            [[self.Model.addConstr(self.G[w,t] >= -(self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            ObjExp = gp.quicksum(self.Data.C[w]*self.G[w,t] for w in self.Data.MS for t in self.Data.Times[1:])
            #Set the objective for minimising the total change in workers, BY SHIFT
        elif Obj=='MinChangeShift':
            self.G=self.Model.addVars(self.Data.MS,self.Data.Times, name='GVar_MinChange',vtype='C')
            for s in self.Data.Shifts:
                T0=self.Data.Shifts[s][0]
                Times=self.Data.Shifts[s][1:]
                
                [self.Model.addConstr(self.G[w,T0] == 0,name="GCons") for w in self.Data.MS]
                [self.Model.addConstr(self.G[w,T0] == 0,name="GCons") for w in self.Data.MS]
                
                [[self.Model.addConstr(self.G[w,t] >= (self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in Times] for w in self.Data.MS]
                [[self.Model.addConstr(self.G[w,t] >= -(self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in Times] for w in self.Data.MS]
                ObjExp = gp.quicksum(self.Data.C[w]*self.G[w,t] for w in self.Data.MS for t in self.Data.Times[1:])
        #Set the objective for minimising the maximum change in the number of workers
        elif Obj=='MinMaxChange':
            self.G=self.Model.addVars(self.Data.MS, name='GVar_MinMax',vtype='C')
            [[self.Model.addConstr(self.G[w] >= (self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            [[self.Model.addConstr(self.G[w] >= -(self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            ObjExp = gp.quicksum(self.G[w] for w in self.Data.MS)
        elif Obj=='DelayMailCost':
            #Create expression for the penalties for delayed flow
            #First, penalties from WAs with 'All' priority
            AllPriorityPenFlows=gp.quicksum(self.X[u,v,k] for (u,v) in self.PenArcs for k in self.Data.Comods if (u,v,k) in self.X)
            #Next penalties from WAs with a partial priority
            PartPriorityPenFlows=gp.quicksum(self.Z.values())
            #Combine these, and multiply by the set DelayFlowPen
            ObjExp=self.DelayFlowPen*(AllPriorityPenFlows+PartPriorityPenFlows)
        #Return the ObjExp
        return ObjExp
    
    def set_objective(self,Obj):
        #Check Obj is one of the acceptable arguments
        assert Obj in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
        ObjExp = self.create_objective(Obj)
        self.Model.setObjective(ObjExp)
        self.Model.update()
        
    def set_lex_objective(self,Objs):
        #Check Obj1 and Obj2 are one of the acceptable arguments
        for o in range(len(Objs)):
            Obj=Objs[o]
            assert Obj in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            print("Obj",o+1,"is",Obj)
            ObjExp=self.create_objective(Obj)
            Priority=len(Objs)-o
            Index=o
            self.Model.setObjectiveN(ObjExp,index=Index,priority=Priority,name=Obj)
        
        self.Model.update()

        
    def set_weighted_objective(self,Objs,Weights):
        #Check there are only as many weights as there are objectives
        assert len(Weights)==len(Objs)
        #Check the weights sum to 1
        assert sum(Weights)==1
        #Create scaling factor for each objective (so they are all roughly same order of magnitude).
        #NOTE: Currently only set for MinCost and MinChange - don't know scales of other ones, so I've put a string, so it'll create an error 
        ScaleFactor={"MinCost":1/10000,
                     "MinChange":1/100,
                     "MinMaxWorker":"a",#1/10,
                     "MinMaxChange":"a",
                     "MinMaxWorkerTime":1/100,
                     "DelayMailCost":1}#1/10}
        #Check each obj is an acceptable arguement, and all weights are >=0. If so, add it
        for i in range(len(Objs)):
            assert Objs[i] in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            assert Weights[i] >=0
            print("Obj ",i+1," is ",Objs[i],", with weight ",Weights[i])
            ObjExp=self.create_objective(Objs[i])
            self.Model.setObjectiveN(ScaleFactor[Objs[i]]*ObjExp,index=i,name=Objs[i],weight=Weights[i])
    
    def set_weighted_objective_Manual(self,Objs,Weights):
        #Check there are only as many weights as there are objectives
        assert len(Weights)==len(Objs)
        #Check the weights sum to 1
        assert sum(Weights)==1
        #Create scaling factor for each objective (so they are all roughly same order of magnitude).
        #NOTE: Currently only set for MinCost and MinChange - don't know scales of other ones, so I've put a string, so it'll create an error 
        ScaleFactor={"MinCost":1/10000,
                     "MinChange":1,#/100,
                     "MinMaxWorker":"a",#1/10,
                     "MinMaxChange":"a",
                     "MinMaxWorkerTime":1/100,
                     "DelayMailCost":1}#1/10}
        #Check each obj is an acceptable arguement, and all weights are >=0. If so, add it
        WeightedObjExp=gp.LinExpr()
        for i in range(len(Objs)):
            assert Objs[i] in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            assert Weights[i] >=0
            print("Obj ",i+1," is ",Objs[i],", with weight ",Weights[i])
            ObjExp=self.create_objective(Objs[i])
            WeightedObjExp+=Weights[i]*(ScaleFactor[Objs[i]]*ObjExp)
        self.Model.setObjective(WeightedObjExp)
        
        
    def Solve(self,Threads=1,TimeLim=300):
        ##Optimise
        self.Model.update()
        self.Model.setParam("Seed",1)
        self.Model.setParam("Threads",Threads)
        self.Model.setParam("TimeLimit",TimeLim)
        #self.Model.setParam("MIPFocus",3)
        self.Model.optimize()
    
    def Solve2(self,Threads=1,TimeLim=300,GapTol=0.01):
        ##Function to optimise, but with variable gap - useful to stop model "getting stuck" at an optimality gap of 1% or so
        print("SOLVE 2 HAPPENING")
        Epoch=300
        self.Model.update()
        self.Model.setParam("Seed",1)
        self.Model.setParam("Threads",Threads)
        if TimeLim <= 2*Epoch:
            self.Model.setParam("TimeLimit",TimeLim)
            #self.Model.setParam("MIPFocus",3)
            self.Model.optimize()
        else:
            #Epoch=300
            Time=0
            CurrGap=0
            PrevGap=1000
            self.Model.setParam("TimeLimit",Epoch)
            while Time < TimeLim:
                print("Time is ",Time)
                Time+=Epoch
                self.Model.optimize()
                if self.Model.Status!=9:
                    print("Breaking due to model status")
                    break
                CurrObj=self.Model.ObjVal
                CurrBound=self.Model.ObjBoundC
                CurrGap=(CurrObj-CurrBound)/CurrObj #self.Model.MIPGap
                GapDiff=PrevGap-CurrGap
                if GapDiff < GapTol:
                    print("PrevGap is ",PrevGap,", CurrGap is ",CurrGap,", Diff is ",GapDiff)
                    print("Breaking due to Gap difference")
                    break
                else:
                    PrevGap=CurrGap
            
            
            
        
    def SolveWithRoundHeur(self,Threads=1,TimeLim=GRB.INFINITY):
        #First, determine if variables are integer. If they are, change to continuous
        VarType=self.Y[0,0].vType
        if VarType=='I':
            self.ChangeToCont()
        #Solve the model
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        #If model has a solution, apply rounding heuristic, change to integer, and solve again
        if self.Model.Status in [2,7,8,9]:
            self.ChangeToInt(RoundHeuristic=True)
            self.Model.update()
            self.Solve(Threads=Threads,TimeLim=TimeLim)
        #Otherwise, just print the model status
        else:
            print("Cont model status is ",self.Model.Status)
        
    def Solve_Lex_Objective_Manual(self,Objs,Threads=1,TimeLim=300,OutputInterim=False, Check=True,Increase=0):
        ##Function to manually solve lex models, with rounding heuristic
        nObjs=len(Objs)
        Outputs={}
        for o in Objs:
            #First, check the objectives are acceptable
            assert o in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            #Now create the objective we need
            ObjExp=self.create_objective(o)
            #Set at the objective
            self.Model.setObjective(ObjExp)
            #Solve
            self.Solve(Threads=Threads,TimeLim=TimeLim)
            #If not the last objective, now add the solution as a constraint to the model
            if o!=Objs[-1]:
                #If we want the interim results, check the model and save these now
                if OutputInterim==True:
                    Outputs[o]=self.extractSolution2()
                    if Check==True:
                        CheckModel(self)
                #Get LHS (objective from first function)
                NewConsLHS=ObjExp
                #Get RHS (Obj val from model)
                NewConsRHS=self.Model.ObjVal
                #Add this as a <= Constr
                self.Model.addConstr(NewConsLHS<=(1+Increase)*NewConsRHS,name="LexCon")
                #Update the model
                self.Model.update()
            #If it is the last objective, pull out solution and check
            else:
              Outputs[o]=self.extractSolution2()
              if Check==True:
                  CheckModel(self)
        return Outputs
    
    def Solve_Lex_Objective_Manual2(self,Objs,Threads=1,TimeLim=300,OutputInterim=False, Check=True,Increase=0,GapTol=0.01):
        ##Function to manually solve lex models, with rounding heuristic
        nObjs=len(Objs)
        Outputs={}
        for o in Objs:
            #First, check the objectives are acceptable
            assert o in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            #Now create the objective we need
            ObjExp=self.create_objective(o)
            #Set at the objective
            self.Model.setObjective(ObjExp)
            #Solve
            self.Solve2(Threads=Threads,TimeLim=TimeLim,GapTol=GapTol)
            #If not the last objective, now add the solution as a constraint to the model
            if o!=Objs[-1]:
                #If we want the interim results, check the model and save these now
                if OutputInterim==True:
                    Outputs[o]=self.extractSolution2()
                    if Check==True:
                        CheckModel(self)
                #Get LHS (objective from first function)
                NewConsLHS=ObjExp
                #Get RHS (Obj val from model)
                NewConsRHS=self.Model.ObjVal
                #Add this as a <= Constr
                self.Model.addConstr(NewConsLHS<=(1+Increase)*NewConsRHS,name="LexCon")
                #Update the model
                self.Model.update()
            #If it is the last objective, pull out solution and check
            else:
              Outputs[o]=self.extractSolution2()
              if Check==True:
                  CheckModel(self)
        return Outputs
    
    def Solve_Lex_Objective_Manual_wCons(self,Objs,SecondObjBound,Threads=1,TimeLim=300,OutputInterim=False, Check=True,Increase=0,GapTol=0.01):
        #Function to solve lex models, adding a constraint to the second objective in the first stage
        Outputs={}
        #In all of these, we'll solve and limit Delayed mail first
        DelayMailObjExp=self.create_objective('DelayMailCost')
        self.Model.setObjective(DelayMailObjExp)
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        DelayMailRHS=self.Model.ObjVal
        self.Model.addConstr(DelayMailObjExp<=DelayMailRHS)
        #For now, assume len(Objs)==2
        assert len(Objs)==2
        #Create expression for 2nd objective, and constrain is
        Obj2Exp=self.create_objective(Objs[1])
        self.Model.addConstr(Obj2Exp <= SecondObjBound,name="FirstStageLexCon")
        #Create first objective, set it, and solve it
        Obj1Exp=self.create_objective(Objs[0])
        self.Model.setObjective(Obj1Exp)
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        Outputs[Objs[0]]=self.extractSolution2()
# =============================================================================
#         #Constrain the first objective
#         NewConsLHS=Obj1Exp
#         #Get RHS (Obj val from model)
#         NewConsRHS=self.Model.ObjVal
#         #Add this as a <= Constr
#         self.Model.addConstr(NewConsLHS<=NewConsRHS,name="LexCon")
#         #Update the model
#         self.Model.update()
#         #Don't need to remove the previous constraint, as we're miniminisng, so it won't be binding
#         #Set the new objective
#         self.Model.setObjective(Obj2Exp)
#         #Solve
#         self.Solve(Threads=Threads,TimeLim=TimeLim)
#         #Extract the new solution
#         Outputs[Objs[1]]=self.extractSolution2()
#         #return the outputs
# =============================================================================
        return Outputs
        
        
    
    def Solve_Lex_Objective_Manual_Soft(self,Objs,Threads=1,TimeLim=300,OutputInterim=False, Check=True,Penalty=10):
        ##Function to manually solve lex models, with rounding heuristic
        Outputs={}
        PenObjective=gp.LinExpr()
        for o in Objs:
            #First, check the objectives are acceptable
            assert o in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','MinMaxWorkerTimeShift','MinChangeShift','DelayMailCost']
            #Now create the objective we need
            ObjExp=self.create_objective(o)
            #Set at the objective
            self.Model.setObjective(ObjExp+PenObjective)
            #Solve
            self.Solve(Threads=Threads,TimeLim=TimeLim)
            #If not the last objective, now add the solution as a constraint to the model
            if o!=Objs[-1]:
                #If we want the interim results, check the model and save these now
                if OutputInterim==True:
                    Outputs[o]=self.extractSolution2()
                    if Check==True:
                        CheckModel(self)
                #Get LHS (objective from first function)
                NewPenObj=ObjExp
                #Get RHS (Obj val from model)
                NewPenTarget=self.Model.ObjVal
                O=self.Model.addVar(vtype='C',name='O_'+o)
                self.Model.addConstr(O>=NewPenObj-NewPenTarget)
                #Update the penalty objective
                PenObjective=PenObjective+Penalty*O
                #Update the model
                self.Model.update()
            #If it is the last objective, pull out solution and check
            else:
              Outputs[o]=self.extractSolution2()
              if Check==True:
                  CheckModel(self)
        return Outputs
        
    def Solve_Lex_MMT_MC_Manual(self,Threads=1,TimeLim=300):
        Objs=['DelayMailCost','MinMaxWorkerTime']
        Sol=self.Solve_Lex_Objective_Manual(Objs,TimeLim=TimeLim,OutputInterim=False,Check=False)
# =============================================================================
#         for o in Objs:
#             #Add expression as a constaint
#             ObjExp=self.create_objective(o)
#             self.Model.setObjective(ObjExp)
#             self.Solve(Threads=Threads,TimeLim=TimeLim)
#             Sol=self.extractSolution2()
#             #Get LHS (objective from first function)
#             NewConsLHS=ObjExp
#             #Get RHS (Obj val from model)
#             NewConsRHS=self.Model.ObjVal
#             #Add this as a <= Constr
#             self.Model.addConstr(NewConsLHS<=NewConsRHS,name="LexCon_"+o)
#             #Update the model
#             self.Model.update()
# =============================================================================
        #Constrain the sum of the number of workers in each work area
        Y=Sol['MinMaxWorkerTime'].Y
        for w in set([w for (w,t) in Y]):
            TRange1=range(0,48)
            TRange2=range(48,96)
            TRange3=range(96,144)
            self.Model.addConstr(gp.quicksum(self.Y[w,t] for t in TRange1)==sum(round(Y[w,t]) for t in TRange1),name="HeurCons_"+str(w)+"_Shift1")
            self.Model.addConstr(gp.quicksum(self.Y[w,t] for t in TRange2)==sum(round(Y[w,t]) for t in TRange2),name="HeurCons_"+str(w)+"_Shift2")
            self.Model.addConstr(gp.quicksum(self.Y[w,t] for t in TRange3)==sum(round(Y[w,t]) for t in TRange3),name="HeurCons_"+str(w)+"_Shift3")
        #Update model
        self.Model.update()
        #Solve for MinMaxWorker
        self.set_objective('MinMaxWorker')
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        #Change to Min Change, and Solve
        self.set_objective('MinChange')
        self.Solve(Threads=Threads,TimeLim=TimeLim)
# =============================================================================
#         Sol2=self.extractSolution2()
#         Outputs['MinChangeRelax']=Sol2.Y
#         return Outputs
# =============================================================================
    

    def Solve_Lex_MMT_MC_SmoothHeur(self,Threads=1,TimeLim=300):
        Objs=['DelayMailCost','MinMaxWorkerTime']
        Sol=self.Solve_Lex_Objective_Manual(Objs,TimeLim=TimeLim,OutputInterim=False,Check=False)
        self.Model.reset()
        for w in set([w for (w,t) in Sol['MinMaxWorkerTime'].Y]):
            for T in [47,95,143]:
                ShiftTimes=range(T-47,T+1)
                YSmall={(w,t):Sol['MinMaxWorkerTime'].Y[w,t] for t in ShiftTimes}
                CapsSmall={(w,t):self.Data.WorkerCap[w,t] for t in ShiftTimes}
                YSmooth=SmoothHeur(YSmall,CapsSmall)
                for (ww,t) in YSmooth:
                    self.Y[ww,t].Start=round(YSmooth[ww,t])
                    #self.Y[ww,t].lb=round(YSmooth[ww,t])
                    #self.Y[ww,t].ub=round(YSmooth[ww,t])
        #Update model
        self.Model.update()
        #Change to Min Change, and Solve
        self.set_objective('MinChange')
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        
    def Solve_MinChange_Seq_Smooth(self,Threads=1,TimeLim=120,WList=range(2,61)):
        ShiftTimes=range(144)
        Gaps={}
        RunTime={}
        Status={}
        self.create_objective('MinChange')
        print(WList)
        for w in WList:
            print("==============  w is ",w,"==============")
            if not all([type(self.Y[w,t])==type(0) for t in ShiftTimes]):
                #self.G=self.Model.addVars(ShiftTimes[1:],vtype='C',name='GVar_Temp')
                #self.G=self.Model.addVars(self.Data.MS,self.Data.Times[1:], name='GVar_MinChange',vtype='C')
                #[self.Model.addConstr(self.G[t] >= (self.Y[w,t]-self.Y[w,t-1]),name='GCons_TempA_'+str(t)) for t in ShiftTimes[1:]]
                #[self.Model.addConstr(self.G[t] >= -(self.Y[w,t]-self.Y[w,t-1]),name='GCons_TempB_'+str(t)) for t in ShiftTimes[1:]]
                
                ObjExp = gp.quicksum(self.Data.C[w]*self.G[w,t] for t in ShiftTimes[1:])
                self.Model.setObjective(ObjExp)
                #self.Model.reset()
                self.Solve(Threads=Threads,TimeLim=TimeLim)
                Gaps[w]=self.Model.MIPGap
                RunTime[w]=self.Model.RunTime
                Status[w]=self.Model.Status
                for t in ShiftTimes:
                    if type(self.Y[w,t])==type(self.Model.getVars()[0]):
                        self.Y[w,t].lb=self.Y[w,t].x
                        self.Y[w,t].ub=self.Y[w,t].x
                    else:
                        assert self.Y[w,t]==0
                #SolTemp=self.extractSolution2()
            else:
                assert sum([self.Y[w,t] for t in ShiftTimes])==0
                print("No workers in work area",w)
            #print("Current Objective is ",SolTemp.CountChanges())
        return [Gaps,RunTime,Status]
    
    def Solve_MinChange_Seq_Smooth_LR(self,PenExp,Target,Iter,uInit,tInit,Threads=1,TimeLim=120,WList=range(2,61)):
        ShiftTimes=range(144)
        Gaps={}
        RunTime={}
        Status={}
        U=[uInit]
        T=[tInit]
        for k in range(Iter):
            u=U[-1]
            t=T[-1]
            for w in WList:
                print("==============  w is ",w,"==============")
                print("Current max is ")
                if not all([type(self.Y[w,t])==type(0) for t in ShiftTimes]):
                    self.G=self.Model.addVars(ShiftTimes[1:],vtype='C',name='GVar_Temp')
                    #self.G=self.Model.addVars(self.Data.MS,self.Data.Times[1:], name='GVar_MinChange',vtype='C')
                    [self.Model.addConstr(self.G[t] >= (self.Y[w,t]-self.Y[w,t-1]),name='GCons_TempA_'+str(t)) for t in ShiftTimes[1:]]
                    [self.Model.addConstr(self.G[t] >= -(self.Y[w,t]-self.Y[w,t-1]),name='GCons_TempB_'+str(t)) for t in ShiftTimes[1:]]
                    
                    ObjExp = gp.quicksum(self.Data.C[w]*self.G[t] for t in ShiftTimes[1:])
                    self.Model.setObjective(ObjExp-u*(Target-PenExp))
                    self.Model.update()
                    self.Solve(Threads=Threads,TimeLim=TimeLim)
                    Gaps[w]=self.Model.MIPGap
                    RunTime[w]=self.Model.RunTime
                    Status[w]=self.Model.Status
                    for t in ShiftTimes:
                        if type(self.Y[w,t])==type(self.Model.getVars()[0]):
                            self.Y[w,t].lb=self.Y[w,t].x
                            self.Y[w,t].ub=self.Y[w,t].x
                        else:
                            assert self.Y[w,t]==0
                    #SolTemp=self.extractSolution2()
                else:
                    assert sum([self.Y[w,t] for t in ShiftTimes])==0
                    print("No workers is work area",w)
            Sol=self.extractSolution2()
            if PenExp.size()>1:
                PenEval=PenExp.getValue()
            else:
                PenEval=PenExp.x
            NewU=max(0,u-t*(Target-PenEval))
            print("Current objective is ",self.Model.ObjVal,", Max workers is ",Sol.FindMax(),", Changes is ",Sol.CountChanges(),", u is ",u)
            U.append(NewU)
            T.append(1/(1+1/t))
                #print("Current Objective is ",SolTemp.CountChanges())
        return [Gaps,RunTime,Status,U,T]
    
    def Solve_by_LR(self,Objs,Threads=1,TimeLim=300,Iter=100,uInit=10,tInit=1):
        #Check objectives are ok
        for o in Objs:
            assert o in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime','DelayMailCost']
        #Set the objective for the first Objective
        ObjExp=self.create_objective(Objs[0])
        self.Model.setObjective(ObjExp)
        #Solve the first Objective
        self.Solve(Threads=Threads,TimeLim=60)
        #Pull out the objective value
        B=self.Model.ObjVal
        #Create the penalty Expression
        PenExp=B-ObjExp
        u=[uInit]
        t=[tInit]
        tol=0.0001
        ObjExp2=self.create_objective(Objs[1])
        print("==================== STARTING LR ====================")
        for k in range(Iter):
            print("Iteration ",k)
            U=u[-1]
            T=t[-1]
            #Set objective with current parameter
            self.Model.setObjective(ObjExp2-U*PenExp)
            self.Model.update()
            #Solve the Model
            self.Solve(Threads=Threads,TimeLim=TimeLim)
            #Pull out the penalty value
            PenVal=PenExp.getValue()
            #Print what's going on
            print(self.Model.ObjVal,u[-1],PenVal)
            #Check to see if we can terminate
            if abs(PenVal)<tol:
                break
            else:
                #Update u
                print(U,T,PenVal)
                u.append(max(0,U+T*(-1*PenVal)))
                #Update t
                t.append(T/2)
        return {"u":u,
                "t":t}
                
            
        
    def Solve_by_Shift(self,Threads=1,TimeLim=300):
        NodeWAs=nx.get_node_attributes(self.Data.DG2,'WS')
        NodeTimes=nx.get_node_attributes(self.Data.DG2,'Time')
        VarType=type(self.Model.getVars()[0])
        #Check Model is already solved
        #assert self.Model.Status in [2,7,8,9]
        #=====SHIFT 1======
        print('=====SHIFT 1======')
        #Fix YVals at their highest point for later shifts
        Wvals=set([w for (w,t) in self.Y])
        for w in Wvals:
            for t in range(48,144):
                if type(self.Y[w,t])==VarType:
                    #print(w,t)
                    self.Y[w,t].lb=self.Y[w,t].x
                    self.Y[w,t].ub=self.Y[w,t].x
        #Fix the X vals
        for (u,v,k) in self.X:
            if type(self.X[u,v,k])==VarType:
                if NodeTimes[u]>=48:
                    self.X[u,v,k].lb=self.X[u,v,k].x
                    self.X[u,v,k].ub=self.X[u,v,k].x
                elif NodeTimes[u]==47 and NodeWAs[u]==NodeWAs[v]:
                    #print(u,v,k)
                    self.X[u,v,k].lb=self.X[u,v,k].x
                    self.X[u,v,k].ub=self.X[u,v,k].x
        #Solve the model
        self.set_objective('MinChange')
        self.Model.update()
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        
        #=====SHIFT 2======
        print('=====SHIFT 2======')
        #Fix the previous shift Yvals and X vals
        for w in Wvals:
            for t in range(48):
                if type(self.Y[w,t])==VarType:
                    self.Y[w,t].lb=self.Y[w,t].x
                    self.Y[w,t].ub=self.Y[w,t].x
        #Fix the X vals
        for (u,v,k) in self.X:
            if type(self.X[u,v,k])==VarType:
                if NodeTimes[u]<48:
                    self.X[u,v,k].lb=self.X[u,v,k].x
                    self.X[u,v,k].ub=self.X[u,v,k].x
        #Unfix the current shift
        for w in Wvals:
            for t in range(48,96):
                if type(self.Y[w,t])==VarType:
                    self.Y[w,t].lb=0
                    self.Y[w,t].ub=self.Data.WorkerCap[w,t]
        for (u,v,k) in self.X:
            if type(self.X[u,v,k])==VarType:
                if NodeTimes[u] in range(48,95):
                    self.X[u,v,k].ub=self.Data.ComCap[u,v][k]
                    self.X[u,v,k].lb=0
                elif NodeTimes[u]==95 and NodeWAs[u]!=NodeWAs[v]:
                    self.X[u,v,k].ub=self.Data.ComCap[u,v][k]
                    self.X[u,v,k].lb=0
        #Following shift already fixed
        #Solve the model
        #self.set_objective('MinChange')
        self.Model.update()
        self.Solve(Threads=Threads,TimeLim=TimeLim)
        #=====SHIFT 3======
        print('=====SHIFT 3======')
        #Fix the previous shift Yvals and X vals
        for w in Wvals:
            for t in range(96):
                if type(self.Y[w,t])==VarType:
                    self.Y[w,t].lb=self.Y[w,t].x
                    self.Y[w,t].ub=self.Y[w,t].x
        #Fix the X vals
        for (u,v,k) in self.X:
            if type(self.X[u,v,k])==VarType:
                if NodeTimes[u]<96:
                    self.X[u,v,k].lb=self.X[u,v,k].x
                    self.X[u,v,k].ub=self.X[u,v,k].x
        #Unfix the current shift
        for w in Wvals:
            for t in range(96,144):
                if type(self.Y[w,t])==VarType:
                    self.Y[w,t].lb=0
                    self.Y[w,t].ub=self.Data.WorkerCap[w,t]
        for (u,v,k) in self.X:
            if type(self.X[u,v,k])==VarType:
                if NodeTimes[u] in range(96,143):
                    self.X[u,v,k].ub=self.Data.ComCap[u,v][k]
                    self.X[u,v,k].lb=0
        #Solve the model
        self.Model.update()
        self.Solve(Threads=Threads,TimeLim=TimeLim)
                
        
    def ChangeToInt(self,RoundHeuristic=False):
        ##Function to take a continuous (solved) model, turn it into an integer model, and use the rounded (up) continuous
        ##Y values as a starting point of the integer model
        #If using rounding heuristic, check model has a solution, then calculate the rounded values
        if RoundHeuristic==True:
            assert self.Model.Status in [2,7,8,9]
            StartY={(w,t):np.ceil(self.Y[w,t].x) for (w,t) in self.Y}
        #Now, go through, change variables to  integers
        for (w,t) in self.Y:
            self.Y[w,t].VType='I'
        #Set the start values (if needed)
        if RoundHeuristic==True:
            for (w,t) in self.Y:
                self.Y[w,t].Start=StartY[w,t]
        #Finally, update the model
        self.Model.update()
        
    def ChangeToCont(self):
        ##Function to take an integer (solved) model, turn it into an continuous model
        #First, check the model has a solution (either optimal, or timed-out)
        #assert self.Model.Status in [2,7,8,9]
        #Now, go through, change variables to continuous, and set the start values
        VarType=type(self.Model.getVars()[0])
        for (w,t) in self.Y:
            if type(self.Y[w,t])==VarType:
                self.Y[w,t].vType='C'
        #Finally, update the model
        self.Model.update()
        
    def RemoveGVars(self,VarName='GVar'):
        V=[v for v in self.Model.getVars() if VarName in v.varName]
        self.Model.remove(V)
        
    def RemoveGCons(self,ConsName='GCons'):
        C=[c for c in self.Model.getConstrs() if ConsName in c.constrName]
        self.Model.remove(C)
        
    def AddBounds(self):
        for w in self.Data.MS:
                for t in self.Data.Times[1:]:
                    self.G[w,t].ub=max(self.Data.WorkerCap[w,t],self.Data.WorkerCap[w,t-1])
                    if type(self.Y[w,t])==type(0):
                        assert self.Y[w,t]==0
                        self.Model.addConstr(self.G[w,t]==self.Y[w,t-1])
                    elif type(self.Y[w,t-1])==type(0):
                        assert self.Y[w,t-1]==0
                        self.Model.addConstr(self.G[w,t]==self.Y[w,t])
# =============================================================================
#                 if 0 in self.Data.WorkPlan[['Early','Late','Night']][self.Data.WorkPlan.WA_Number==w].values:
#                     GSumExp=gp.quicksum(self.G[w,t] for t in self.Data.Times[1:])
#                     for t in self.Data.Times:
#                         self.Model.addConstr(GSumExp>=self.Y[w,t])
# =============================================================================
# =============================================================================
#                 GSumExp=gp.quicksum(self.G[w,t] for t in self.Data.Times[1:])
#                 for (t1,t2) in itertools.combinations(self.Data.Times,2):
#                     self.Model.addConstr(GSumExp>=(self.Y[w,t1]-self.Y[w,t2]))
#                     self.Model.addConstr(GSumExp>=-(self.Y[w,t1]-self.Y[w,t2]))
# =============================================================================
        
    def extractSolution(self):
        #Check the model is either infeasible or solved, or reached a user-set limit
        assert self.Model.status in [2,3,7,8,9]
        print("Model status is ",self.Model.status)
        if self.Model.status!=3:
            XVars=dict(self.Model.getAttr('x',self.X))
            YVars=dict(self.Model.getAttr('x',self.Y))
            #Extract the G vars
            #Firstly, are there no G vars (for min cost objective)
            if self.G==None:
                return XVars, YVars
            else:
                #Secondly, are there multiple G vars (for most objectives)
                if type(self.G)==type(self.Y):
                    GVars=dict(self.Model.getAttr('x',self.G))
                #Thirdly, is there only one G objective (for )
                else:
                    GVars=self.G.x
                return XVars, YVars, GVars
        else:
            print("Model not solved. Model Status is ", self.Model.status)
            
    def extractSolution2(self):
        #Check the model is either infeasible or solved, or reached a user-set limit
        assert self.Model.status in [2,3,7,8,9]
        print("Model status is ",self.Model.status)
        if self.Model.status!=3:
            return DetModelSolution(self)
        else:
            print("Model not solved. Model Status is ", self.Model.status)
            
            


class DetModelSolution():
    
    def __init__(self,DetModel):
        #self.X=dict(DetModel.Model.getAttr('x',DetModel.X))
        VarType=type(DetModel.Model.getVars()[0])
        self.X={k:DetModel.X[k].x if type(DetModel.X[k])==VarType else DetModel.X[k] for k in DetModel.X}
        #self.Y=dict(DetModel.Model.getAttr('x',DetModel.Y))
        self.Y={k:DetModel.Y[k].x if type(DetModel.Y[k])==VarType else DetModel.Y[k] for k in DetModel.Y}
        self.Z=dict(DetModel.Model.getAttr('x',DetModel.Z))
        self.H=dict(DetModel.Model.getAttr('x',DetModel.H))
        self.A=dict(DetModel.Model.getAttr('x',DetModel.A))
        if DetModel.G == None:
            self.G=None
        elif type(DetModel.G)==type(DetModel.Z):
            self.G=dict(DetModel.Model.getAttr('x',DetModel.G))
        elif type(DetModel.G)==VarType:
            self.G=DetModel.G.x
        else:
            print("Weird G type")
            assert 1==2
        self.C = DetModel.Data.C
        self.WAList=DetModel.Data.MS
        self.WorkPlanDF=pd.DataFrame([[self.Y[w,t] for w in self.WAList] for t in DetModel.Data.Times])
        self.DG2 = DetModel.Data.DG2
        self.Times=DetModel.Data.Times
        self.CommodsList = DetModel.Data.Comods
        T=nx.get_node_attributes(self.DG2,'Time')
        W=nx.get_node_attributes(self.DG2,'WS')
        self.PlaceTimeDict={(W[n],T[n]):n for n in W}
        self.PenArcs = DetModel.PenArcs
        self.DelayFlowPen = DetModel.DelayFlowPen
        try:
            DetModel.Model.MIPGap
        except AttributeError:
            self.ModStats={'Status':DetModel.Model.Status,
                           'Time':DetModel.Model.RunTime,
                           'MIPGap':'NA - No MIPGap'}
        else:
            self.ModStats={'Status':DetModel.Model.Status,
                           'Time':DetModel.Model.RunTime,
                           'MIPGap':DetModel.Model.MIPGap}
        
    def AllPriorityArcCost(self):
        return sum(self.X[u,v,k] for k in self.CommodsList for (u,v) in self.PenArcs if (u,v,k) in self.X)
    
    def TargetPriorityArcCost(self):
        return sum(self.Z.values())
    
    def TotalArcCost(self,Penalty=None):
        if Penalty==None:
            Penalty=self.DelayFlowPen
        return Penalty*(self.AllPriorityArcCost()+self.TargetPriorityArcCost())
        
        
    def WorkerCost(self):
        '''
        Returns the total cost of the workers in a given solution

        Returns
        -------
        TYPE Int
            DESCRIPTION. The total cost of the workers

        '''
        return sum([self.C[w]*self.Y[w,t] for (w,t) in self.Y])
    
    def CountChanges(self):
        TotalChange=0
        for w in self.WAList:
            TotalChange+=sum(np.abs([self.C[w]*(self.Y[w,t]-self.Y[w,t-1]) for t in self.Times[1:]]))
        return TotalChange
    
    def FindMax(self):
        Workers=[sum([self.C[w]*self.Y[w,t] for w in self.WAList]) for t in self.Times]
        return np.max(Workers)
    
    def DelayedMail(self):
        #EdgeTypes=nx.get_edge_attributes(self.DG2,'Type')
        #DelayEdges=[e for (e,atts) in self.Data.DG2.edges.items() if atts['Type']=='Delay']
        #CompEdges=[e for (e,atts) in self.Data.DG2.edges.items() if atts['Type']=='Sink']
        DelayedMail={w:sum(self.X[self.PlaceTimeDict[w,self.Times[-1]],k] for k in self.CommodsList if (self.PlaceTimeDict[w,self.Times[-1]],k) in self.X) for w in self.WAList}
        return DelayedMail
        
    
    def SortedMail(self):
        '''
        Function to calculate the amount of mail sorted at each work station at each time.

        Returns
        -------
        Outputs : TYPE Dictionary
            DESCRIPTION. Dictionary of sorted mail at each work area at each time. Keys are (w,t), values are amount of sorted mail. CURRENTLY TOTAL MAIL, NOT SPLIT BY COMMODITY

        '''
        E=nx.get_edge_attributes(self.DG2,'Type')
        Outputs={(w,t):sum(self.X[e[0],e[1], k] for k in self.CommodsList for e in self.DG2.out_edges(self.PlaceTimeDict[(w,t)]) if E[e] not in ['Hold','Delay']) for (w,t) in self.PlaceTimeDict}
        return Outputs
        
    
    def SpareCapacity(self, Throughputs):
        '''
        Calculate the spare capacity of a solution

        Parameters
        ----------
        Throughputs : TYPE Dictionary
            DESCRIPTION. Dictionary of Throughputs. Keys are work areas, values are throughputs (per worker)

        Returns
        -------
        TYPE Int
            DESCRIPTION. Dictionary of work areas and the spare capacity in each time period. Keys are tuples of work area and time, values are spare capacity

        '''
        Capacities = {(w,t):self.Y[w,t]*Throughputs[w] for (w,t) in self.Y}
        
        SpareCaps={(w,t):Capacities[w,t]-self.SortedMail()[w,t] for (w,t) in Capacities}
        
        return SpareCaps
    
    def WorkerDF(self):
        return pd.DataFrame([[self.Y[w,t] for w in self.WAList] for t in self.Times])
        
        
        
    

def SmoothHeur(Y,Caps):
    K=list(Y.keys())
    w=K[0][0]
    Times=[k[1] for k in K]
    m=gp.Model()
    mY=m.addVars(Y,ub=Caps,vtype='I')
    G=m.addVars(Times[1:],vtype='C')
    #G=m.addVar(vtype='C')
    m.addConstr(gp.quicksum(mY)==sum(Y.values()))
    for t in Times[1:]:
        m.addConstr((mY[w,t]-mY[w,t-1])<=G[t])
        m.addConstr(-(mY[w,t]-mY[w,t-1])<=G[t])
    #m.addConstrs(G<=mY[w,t] for (w,t) in Y)
    m.setObjective(gp.quicksum(G))
    #m.setObjective(G)
    m.setParam("OutputFlag",0)
    m.update()
    m.optimize()
    YVars=m.getAttr('x',mY)
    return YVars
    


# =============================================================================
#     #Get dictionary of work stations for each node
#     WS=nx.get_node_attributes(self.Data.self.Data.DG2,name='WS')
#     print('w',WS[0])
#     
#     
#     
#     
#     self.G = self.Model.addVars(self.Data.MS, name='MinMax',vtype='C')
# =============================================================================
    
    
    
    
    
    


