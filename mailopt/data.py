from dataclasses import dataclass, field
import networkx
import pandas
from typing import List, Any, Tuple, Mapping, Sequence, Optional
# from collections.abc import Mapping, Sequence

TimeNode = Tuple[str,int]
TimeEdge = Tuple[TimeNode, TimeNode]

@dataclass
class ProblemData:
  """Class to represent required data for mail centre optimization problem.

  Attributes:
     DG2 (networkx.DiGraph): The directed time-expanded network you wish to solve - an output of function TimeExpand
     Times (Sequence[int]) : Set of time periods to be considered.
     ComCap (Mapping[TimeEdge, Mapping[str, float]]):
        A dictionary of the capacities for each different commodity on each edge of the network
     TotalCap (Mapping[Any, float]): Total capacity on each edge of the network
     WorkerCap (Mapping[(int,int),int]): Maps each work area and time to the capacity of that work area at that time.
     Comods (Sequence[str]): A list of the names of all the different types of commodities
     MS (Sequence[str]): Set of work areas are manual work areas instead of mechanical
     C (Mapping[str,float]): Specifies cost of one worker for each area of `MS`
     NodeCaps (Mapping[Any, float]): Total capacity coming out of each node of the network
     IDDicts (Optional[Mapping[str, Any]]): Dictionary giving the indirect mappings of the network. Keys are work area names. Values are dictionarys, giving the origin WA, list of destination WA names, and ratios going to each dictionary
     ComodGroups (Optional[Mapping[str, int]]): Maps each commodity to an index of a group. By default each commodity is in its own group.
     Tethered (Optional[Sequence[(int,int)]]): List of tethered work area pairs. Default is []
  """
  DG2: networkx.DiGraph
  Times: List[int]
  ComCap: Mapping[TimeEdge, Mapping[str, float]]
  WorkPlan: pandas.DataFrame
  WorkerCap: Mapping[Tuple,int] #List[int]
  Comods: Sequence[str]
  MS: Sequence[str]
  C: Mapping[str, float]
  NodeCaps: Mapping[Any, float]
  BaseNetwork: networkx.DiGraph
  IDDicts: Optional[Mapping[str, Any]] = field(default_factory=dict)
  ComodGroups: Optional[Mapping[str, int]] = None
  Tethered: Optional[Sequence[Tuple]] = field(default_factory=lambda:[]) #Sequence[(int,int)] = []
  StreamPaths: Optional[Mapping[str,Any]] = None
  WANameNumber: Optional[Mapping[str, int]] = None
  WANumberName: Optional[Mapping[int, str]] = None
  AcceptPaths: Optional[Mapping[str, Tuple]] = None
  Date: Optional[str] = None
  Sources: Optional[List[str]] = None
  Sinks: Optional[List[str]] = None
  Shifts: Optional[Mapping[int, Tuple]] = None
  
  
