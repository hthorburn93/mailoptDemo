# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:17:25 2022

@author: thorburh
"""


from mailopt.data import ProblemData
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mailopt.NetworkBuilding.CreateSmallInstance import CreateSmallNetworkInstance
from mailopt.Deterministic.ModelChecks import CheckModel
from mailopt.Graphing.GraphDrawing import PlotDiGraph
import networkx as nx
import itertools

class StochMCModel:
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
    def __init__(self, CentreData,Demands,VTypes={"X":'C',"Y":'C'},nScenarios=10,Yvals=None,DelayFlowPen=100):
        
        print("Start to build the model")
        self.Data=CentreData
        #print("1")
        self.Model=gp.Model()
        #print("2")
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
        self.Scenarios=range(nScenarios)
        print('Source A',[Demands[z]['Source 0'][0] for z in self.Scenarios])
        print('Sink A',[Demands[z]['Sink 215'][0] for z in self.Scenarios])
        
        WS=nx.get_node_attributes(self.Data.DG2,name='WS')
        NodeTimes=nx.get_node_attributes(self.Data.DG2,name='Time')
        PlaceTimeDict={(WS[n],NodeTimes[n]):n for n in WS}
        PlaceTimeDict2={n:(WS[n],NodeTimes[n]) for n in WS}
        
        
        
        print("Add X variables")
        # Add X variables - X[u,v,k] is how much of commodity k to send along edge (u,v)
        self.X={}
        for (u,v) in self.Data.DG2.edges:
            for k in self.Data.Comods:
                for z in range(nScenarios):
                    if WS[u]==WS[v]:
                        VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)+'_'+str(z)
                        self.X[u,v,k,z]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                    elif WS[v]=='Sink':
                        VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)+'_'+str(z)
                        self.X[u,v,k,z]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                    else:
                        if self.Data.ComCap[(u,v)][k]==0:
                            self.X[u,v,k,z]=0
                        elif self.Data.NodeCaps[u]==0:
                            self.X[u,v,k,z]=0
                        elif PlaceTimeDict2[u] in self.Data.WorkerCap and self.Data.WorkerCap[PlaceTimeDict2[u]]==0:
                            self.X[u,v,k,z]=0
                        elif self.Data.ComCap[(u,v)][k]>0:
                            VarName='Flow_'+str(WS[u])+'_'+str(WS[v])+'_'+str(NodeTimes[u])+'_'+str(k)+'_'+str(z)
                            self.X[u,v,k,z]=self.Model.addVar(name=VarName,vtype=self.VTypes['X'])
                        else:
                            print("Missing X val category for ",(u,v,k))
                            assert 1==2
        #self.X = self.Model.addVars(self.Data.DG2.edges, self.Data.Comods, name='Flow', vtype=self.VTypes['X'])
        
        #print("Yvals is ",Yvals)
        if Yvals!=None:
            print("Y variables given")
            self.Y=Yvals
        else:
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
        #Get a type for a variable
        VarType=type(self.Model.getVars()[0])
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
            #if attrs['WS'] in ['Source','Sink']:
                #print(attrs['WS'],[Demands[z][n][0] for z in self.Scenarios])
            self.Model.addConstrs((gp.quicksum(self.X[u,v,k,z] for (u,v) in self.Data.DG2.out_edges(n) if (u,v,k,z) in self.X) - gp.quicksum(self.X[u,v,k,z] for (u,v) in self.Data.DG2.in_edges(n) if (u,v,k,z) in self.X) == Demands[z][n][k]
                          for  k in self.Data.Comods for z in self.Scenarios), name='MassBal_'+str(attrs['WS'])+"_"+str(attrs['Time']))
        
        print("Add Arc Capacity Constraints")
        ## Arc Capacity Constraints
        for (e, e_attrs) in self.Data.DG2.edges.items():
            for z in self.Scenarios:
                u,v=e
                #If any flow is allowed on the arc, add the commodity capacities
                if self.Data.NodeCaps[u]>0:
                    ComList4Edge=[k for k in self.Data.Comods if (u,v,k,z) in self.X]
                    #print(u,v,ComList4Edge)
    # =============================================================================
    #                 for k in ComList4Edge:
    #                     print(u,v,k,self.X[u,v,k],self.Data.ComCap[u,v][k],type(self.X[u,v,k]),type(self.Data.ComCap[u,v][k]),self.X[u,v,k] <= self.Data.ComCap[u,v][k])
    #                     if type(self.X[u,v,k])==VarType:
    #                         self.Model.addConstr(self.X[u,v,k] <= self.Data.ComCap[u,v][k],name="ComCap")
    # =============================================================================
                    self.Model.addConstrs((self.X[u,v,k,z] <= self.Data.ComCap[u,v][k] for k in ComList4Edge if type(self.X[u,v,k,z])==VarType),name="ComCap")
        ## Add total workflow capacity constraints
        # That is, ALL edges coming from a node (which aren't the hold edge) need to be constrained by the throughput
        for (n,attrs) in self.Data.DG2.nodes.items():
            for z in self.Scenarios:
                ExitEdges=[(u,v) for (u,v,Type) in self.Data.DG2.out_edges(n,data='Type') if Type not in ["Hold","Delay"]]
                w=WS[n]
                t=NodeTimes[n]
                if attrs['WS'] not in self.Data.MS+['Source','Sink']:
                    self.Model.addConstr(gp.quicksum(self.X[u,v,k,z] for (u,v) in ExitEdges for k in self.Data.Comods if (u,v,k,z) in self.X)<=self.Data.NodeCaps[n],name="NodeCap")
                elif attrs['WS'] in self.Data.MS:
                    WhichWS=self.Data.DG2.nodes[n]['WS']
                    WhichTime=self.Data.DG2.nodes[n]['Time']
                    #print(ExitEdges)
                    if n==4:
                        print("HERE!!!!!")
                    self.Model.addConstr(gp.quicksum(self.X[u,v,k,z] for (u,v) in ExitEdges for k in self.Data.Comods if (u,v,k,z) in self.X)<=self.Y[WhichWS,WhichTime]*self.Data.NodeCaps[n],name="NodeCap_"+str(n)+"_"+str(z))
                else:
                    if WS[n] not in ['Source','Sink']:
                        print("N is ",n)
                        assert 1==2
        print("Add ID stream constraints")
        #Add indirect stream constraints
        for i in self.Data.IDDicts:
            for z in self.Scenarios:
                InFlows=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[v]==self.Data.WANameNumber[i] and Type!='Hold']
                #print("i is ",i,", len(InFlows) is ",len(InFlows))
                InFlowSums={k:gp.quicksum(self.X[u,v,k,z] for (u,v) in InFlows if (u,v,k,z) in self.X) for k in self.Data.Comods}
                DelayEdges=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[u]==self.Data.WANameNumber[i] and Type=='Delay']
                #print(DelayEdges)
                assert len(DelayEdges)==1
                DelayEdge=DelayEdges[0]
                for d in range(len(self.Data.IDDicts[i]['Destinations'])):
                    DestWA=self.Data.IDDicts[i]['Destinations'][d]
                    DestRatio=self.Data.IDDicts[i]['Ratios'][d]
                    OutFlows=[(u,v) for (u,v,Type) in self.Data.DG2.edges(data='Type') if WS[u]==self.Data.WANameNumber[i] and WS[v]==self.Data.WANameNumber[DestWA]]
                    #print("i is ",i,", len(OutFlows) is ",len(OutFlows), ", DestWA is ", DestWA,', DestRatio is ',DestRatio)
                    OutFlowSums={k:gp.quicksum(self.X[u,v,k,z] for (u,v) in OutFlows if (u,v,k,z) in self.X) for k in self.Data.Comods}
                    for k in self.Data.Comods:
                        if (DelayEdge[0],DelayEdge[1],k) in self.X:
                            #print("Flow Exists",(DelayEdge[0],DelayEdge[1],k))
                            self.Model.addConstr(OutFlowSums[k]==DestRatio*(InFlowSums[k]-self.X[DelayEdge[0],DelayEdge[1],k,z]),name='ID'+str(i)+str(DestWA)+str(k)+str(z))
                        else:
                            #print("Flow Doesn't Exist",(DelayEdge[0],DelayEdge[1],k))
                            self.Model.addConstr(OutFlowSums[k]==DestRatio*(InFlowSums[k]),name='ID'+str(i)+str(DestWA)+str(k)+str(z))
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
        ShiftEnds={'Early':7,
                   'Late':15,
                   'Night':23}
        #Now, loop through the WorkPlan WA Numbers and Shifts
        for w in self.Data.WorkPlan['WA_Number'].values:
            for Shift in ['Early','Late','Night']:
                Priority=self.Data.WorkPlan[self.Data.WorkPlan.WA_Number==w][Shift].values[0]
                t=ShiftEnds[Shift]
                #If Priority is 0, just assert that upper bound on Y is 0
                if Priority==0:
                    ShiftStartTime=t-8+1
                    for tt in range(ShiftStartTime,t+1):
                        if Yvals==None:
                            assert self.Y[w,tt]==0       
                #If Priority is all, find the edge for the hold/delay arc for this shift, and add it to the PenArcs List
                elif Priority=='All':
                    if w in set(WS.values()):
                        Origin=PlaceTimeDict[w,t]
                        if t==23:
                            Dest=PlaceTimeDict['Sink',24]
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
                    ShiftStart=t-8+1
                    #Get a list of all edges (which aren't holding edges) which send flow into the WA
                    EntryEdges=[(u,v) for tt in range(ShiftStart,t+1) for (u,v) in self.Data.DG2.in_edges(PlaceTimeDict[w,tt]) if WS[u]!=WS[v]]
                    #If it's not the first shift, also include the holding edge from the previous shift (for the leftover mail)
                    if t!= 7:
                        EntryEdges+=[(PlaceTimeDict[w,ShiftStart-1],PlaceTimeDict[w,ShiftStart])]
                    #Get a list of all edges (which aren't holding edges) which send flow out of the WA (but not to the sink - only relevant for the final shift)
                    ExitEdges=[(u,v) for tt in range(ShiftStart,t+1) for (u,v) in self.Data.DG2.out_edges(PlaceTimeDict[w,tt]) if WS[u]!=WS[v] and WS[v]!='Sink']
                    #Using these edges, calculate the flows into and out of the work area
                    #FlowIn=gp.quicksum(self.X[u,v,k] for (u,v) in EntryEdges for k in self.Data.Comods if (u,v,k) in self.X)
                    FlowOut=gp.quicksum(self.X[u,v,k,z] for (u,v) in ExitEdges for k in self.Data.Comods for z in self.Scenarios if (u,v,k,z) in self.X)
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
        
    def create_objective_1stStage(self,Obj):
        #Check Obj is one of the acceptable arguments
        assert Obj in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime']
        
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
        #Set the objective for minimising the total change in workers
        elif Obj=='MinChange':
            self.G=self.Model.addVars(self.Data.MS,self.Data.Times[1:], name='GVar_MinChange',vtype='C')
            [[self.Model.addConstr(self.G[w,t] >= (self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            [[self.Model.addConstr(self.G[w,t] >= -(self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            ObjExp = gp.quicksum(self.Data.C[w]*self.G[w,t] for w in self.Data.MS for t in self.Data.Times[1:])
        #Set the objective for minimising the maximum change in the number of workers
        elif Obj=='MinMaxChange':
            self.G=self.Model.addVars(self.Data.MS, name='GVar_MinMax',vtype='C')
            [[self.Model.addConstr(self.G[w] >= (self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            [[self.Model.addConstr(self.G[w] >= -(self.Y[w,t]-self.Y[w,t-1]),name="GCons") for t in self.Data.Times[1:]] for w in self.Data.MS]
            ObjExp = gp.quicksum(self.G[w] for w in self.Data.MS)
        
        #Return the ObjExp
        return ObjExp
    
    def create_objective_2ndStage(self,Obj):
        #Check Obj is one of the acceptable arguments
        assert Obj in ['DelayMailCost']
        
        if Obj=='DelayMailCost':
            #Create expression for the penalties for delayed flow
            #First, penalties from WAs with 'All' priority
            AllPriorityPenFlows=gp.quicksum(self.X[u,v,k,z] for (u,v) in self.PenArcs for k in self.Data.Comods for z in self.Scenarios if (u,v,k,z) in self.X)
            #print("AllPriorityPenFlows is ", AllPriorityPenFlows)
            #Next penalties from WAs with a partial priority
            PartPriorityPenFlows=gp.quicksum(self.Z.values())
            #print("PartPriorityPenFlows is ", PartPriorityPenFlows)
            #Combine these, and multiply by the set DelayFlowPen
            ObjExp=(1/len(self.Scenarios))*self.DelayFlowPen*(AllPriorityPenFlows+PartPriorityPenFlows)
            #print("Create Obj ",ObjExp)
            
        #Return the ObjExp
        return ObjExp
    
    def set_objective(self,Obj1,Obj2):
        #Check Obj is one of the acceptable arguments
        assert Obj1 in ['MinCost','MinMaxWorker','MinChange','MinMaxChange','MinMaxWorkerTime']
        assert Obj2 in ['DelayMailCost']
        ObjExp1stStage = self.create_objective_1stStage(Obj1)
        #print(ObjExp1stStage)
        ObjExp2ndStage = self.create_objective_2ndStage(Obj2)
        #print(ObjExp2ndStage)
        self.Model.setObjective(ObjExp1stStage+ObjExp2ndStage)
        self.Model.update()
        
    def Solve(self,Threads=1,TimeLim=300):
        ##Optimise
        self.Model.update()
        self.Model.setParam("Seed",1)
        self.Model.setParam("Threads",Threads)
        self.Model.setParam("TimeLimit",TimeLim)
        #self.Model.setParam("MIPFocus",3)
        self.Model.optimize()
        
    def extractSolution(self):
        #Check the model is either infeasible or solved, or reached a user-set limit
        assert self.Model.status in [2,3,7,8,9]
        print("Model status is ",self.Model.status)
        if self.Model.status!=3:
            return StochModelSolution(self)
        else:
            print("Model not solved. Model Status is ", self.Model.status)
    def PlotDiGraph(self):
        #PlotDiGraph(Graph,UniqueWS,FigSize=(20,10),width=1,labels=None,NodeCols='blue')
        Widths=[self.Data.ComCap[u,v][0]>0 for (u,v) in self.Data.DG2.edges()]
        PlotDiGraph(self.Data.DG2,['Source']+self.Data.MS+['Completion','Sink'],width=Widths,labels=None,NodeCols='blue')


class StochModelSolution():
    
    def __init__(self,StochModel):
        #self.X=dict(StochModel.Model.getAttr('x',StochModel.X))
        VarType=type(StochModel.Model.getVars()[0])
        self.X={k:StochModel.X[k].x if type(StochModel.X[k])==VarType else StochModel.X[k] for k in StochModel.X}
        #self.Y=dict(StochModel.Model.getAttr('x',StochModel.Y))
        self.Y={k:StochModel.Y[k].x if type(StochModel.Y[k])==VarType else StochModel.Y[k] for k in StochModel.Y}
        self.Z=dict(StochModel.Model.getAttr('x',StochModel.Z))
        self.H=dict(StochModel.Model.getAttr('x',StochModel.H))
        self.A=dict(StochModel.Model.getAttr('x',StochModel.A))
        if StochModel.G == None:
            self.G=None
        elif type(StochModel.G)==type(StochModel.Z):
            self.G=dict(StochModel.Model.getAttr('x',StochModel.G))
        elif type(StochModel.G)==VarType:
            self.G=StochModel.G.x
        else:
            print("Weird G type")
            assert 1==2
        self.C = StochModel.Data.C
        self.WAList=StochModel.Data.MS
        self.WorkPlanDF=pd.DataFrame([[self.Y[w,t] for w in self.WAList] for t in StochModel.Data.Times])
        self.DG2 = StochModel.Data.DG2
        self.Times=StochModel.Data.Times
        self.CommodsList = StochModel.Data.Comods
        T=nx.get_node_attributes(self.DG2,'Time')
        W=nx.get_node_attributes(self.DG2,'WS')
        self.PlaceTimeDict={(W[n],T[n]):n for n in W}
        self.PenArcs = StochModel.PenArcs
        self.DelayFlowPen = StochModel.DelayFlowPen
        self.Scenarios = StochModel.Scenarios
        try:
            StochModel.Model.MIPGap
        except AttributeError:
            self.ModStats={'Status':StochModel.Model.Status,
                           'Time':StochModel.Model.RunTime,
                           'MIPGap':'NA - No MIPGap'}
        else:
            self.ModStats={'Status':StochModel.Model.Status,
                           'Time':StochModel.Model.RunTime,
                           'MIPGap':StochModel.Model.MIPGap}
        
    def AllPriorityArcCost(self):
        return sum(self.X[u,v,k,z] for k in self.CommodsList for (u,v) in self.PenArcs for z in self.Scenarios if (u,v,k,z) in self.X)
    
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
    
    def PlotFlows(self):
        #PlotDiGraph(Graph,UniqueWS,FigSize=(20,10),width=1,labels=None,NodeCols='blue')
        Widths=[sum([self.X[u,v,k,z] for k in self.CommodsList for z in self.Scenarios])>0.000001 for (u,v) in self.DG2.edges()]
        PlotDiGraph(self.DG2,['Source']+self.WAList+['Completion','Sink'],width=Widths,labels=None,NodeCols='blue')