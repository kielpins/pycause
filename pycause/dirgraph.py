"""
Define DirGraph extension of networkx.DiGraph class, just adding some sugar.
"""

import networkx as nx

class DirGraph(nx.DiGraph):
    """
    Extension of networkx.DiGraph class. Just adds syntactic sugar for determining graph properties.
    """

    def __init__(self, data=None, **attr):
        """Initialize DirGraph in exactly the same way as parent networkx.DiGraph class. See parent class documentation.
        
        :param data: input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a NumPy matrix
            or 2d ndarray, a SciPy sparse matrix, or a PyGraphviz graph.
        :param attr:  keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.
        """

        nx.DiGraph.__init__(self, data, **attr)
    
    def is_dir_edge(self, edge):
        """Check whether edge exists and is directed.
        
        :param edge: pair of nodes in [parent, child] order
        :return: True if edge exists and is directed, False otherwise
        """

        if edge in self.edges() and edge[::-1] not in self.edges():
            return True
        else:
            return False

    def is_undir_edge(self, edge):
        """Check whether edge exists and is undirected.
        
        :param edge: pair of nodes in [parent, child] order
        :return: True if edge exists and is directed, False otherwise
        """

        if edge in self.edges() and edge[::-1] in self.edges():
            return True
        else:
            return False

    def is_adj(self, edge):
        """Check whether a pair of nodes is adjacent.
        
        :param edge: pair of nodes
        :return: True if nodes are adjacent, False otherwise
        """

        if edge in self.edges() or edge[::-1] in self.edges():
            return True
        else:
            return False

    def is_non_adj(self, edge):
        """Check whether a pair of nodes is nonadjacent.
        
        :param edge: pair of nodes
        :return: True if nodes are distinct and nonadjacent, False otherwise
        """

        if not edge in self.edges() and not edge[::-1] in self.edges() and not edge[0]==edge[1]:
            return True
        else:
            return False
