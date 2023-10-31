# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:43:48 2022

@author: thorburh
"""

import numpy as np


def GenerateCommods(nCommods,nSamples=1,Means=None, Var=None):
    """
    

    Parameters
    ----------
    nCommods : TYPE
        DESCRIPTION.
    nSamples : TYPE, optional
        DESCRIPTION. The default is 1.
    Means : TYPE, optional
        DESCRIPTION. The default is None.
    Var : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    CommodNames : TYPE List
        DESCRIPTION. List of commodities
    Demands : TYPE np.array
        DESCRIPTION. np.array of demands for each quantity

    """
    
    #Check that:
        #1. If multiple commods, means and Cov given correctly
        #2. If single, Means given as a number, not a list
    #print("Means 2 a is ",Means)
    if nCommods>1:
        if Means is None:
            #print("YES")
            Means=np.zeros(nCommods)
        if Var is None:
            Var=np.eye(nCommods)
        assert len(Means)==nCommods
        #print(Var.shape)
        assert Var.shape[0]==Var.shape[1]
        assert Var.shape[0]==nCommods
    else:
        assert type(Means)==type(1) or type(Means)==type(1.4)
        assert type(Means)==type(Var)
    
    #Set the commod names
    CommodNames=list(range(nCommods))
    
    #Generate the demands
    if nCommods>1:
        #print("Means 2 is ",Means)
        Demands=np.random.multivariate_normal(mean=Means, cov=Var,size=nSamples)
    else:
        Std=np.sqrt(Var)
        Demands=np.random.normal(loc=Means,scale=Std,size=nSamples)
    
    ##Take absolute value of demands so that the model is always feasible
    Demands=np.abs(Demands)
    
    #Return things
    return CommodNames, Demands


def HKWMomentMatching(n,s,TargetR,TargetMom):
    """
    FUnction to generate multivariate scenarios with target correlation and moments

    Parameters
    ----------
    n : TYPE int
        DESCRIPTION. Number of random variables to generate per scenario
    s : TYPE int
        DESCRIPTION. Number of scenarios to generate
    TargetR : TYPE numpy matrix (n x n)
        DESCRIPTION. Target/desired correlation matrix of the given scenarios
    TargetMom : TYPE numpy matrix (4 x n)
        DESCRIPTION. Target/desired first 4 moments of the variables

    Returns
    -------
    None.

    """
    
    #Step 1 - specify the inputs. That's done already
    
    #Step 2 - Get moments for initial variables to generate
    Mom1=[0 for tm in TargetMom[0]]
    Mom2=[1 for tm in TargetMom[1]]
    #alpha is the square root of the second target moment
    alpha=np.array([tm**0.5 for tm in TargetMom[1]])
    #beta is the first target moment
    beta=np.array([tm for tm in TargetMom[0]])
    #Initial variable moment 3 is target moment 3 divided by alpha^3
    Mom3=[TargetMom[2][i]/(alpha[i]**3) for i in range(len(alpha))]
    Mom4=[TargetMom[3][i]/(alpha[i]**4) for i in range(len(alpha))]
    
    #Get cholesky decomposition of R
    L=np.linalg.cholesky(TargetR)
    
    #Step 3 - transform the moments. The update says this step is redundant
    
    #Step 4 - sample X from a standard normal distribution - NOTE: THIS IS BASED ON THE UPDATE TO THE PAPER (from Kauts website), NOT THE PAPER ITSELF
    X=np.random.multivariate_normal(np.zeros(n),np.eye(n),size=s)
    X=X.T
    
    #Step 5 - transform these back to y
    Y=np.matmul(L,X)
    
    #Step 6 - transform these moments to 
    #Step 6 a - multiply the appropriate alphas
    ZTemp=np.array([np.multiply(alpha,Y.T[i]) for i in range(s)]).transpose()
    
    Z=ZTemp+np.array([beta for i in range(s)]).transpose()
    
    return Z