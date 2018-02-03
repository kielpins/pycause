"""
Functions that return test sets for conditional independence tests.
[1] D. Colombo's PhD thesis (ETH Zurich #21445, 2013), Colombo-causal-learning-high-dimension-THESIS.pdf,
"""

def get_adjs(G, edge):
    """Find neighbors of a node in a graph, excluding a specified edge.
    
    TODO 2: rename this function more appropriately
    
    :param G: graph
    :param edge: pair of nodes
    :return: list of neighbors of first node, excluding second node
    """
    
    x, y = edge
    return [z for z in G.neighbors(x) if z!=y]

def get_fci_sep(G, edge):
    """Find possible d-separation sets for final skeleton of FCI algorithm using procedure defined in [1], p. 30.
    
    :param G: graph
    :param edge: pair of nodes to test for d-separation
    :return: cand_dict = dictionary of candidate d-separation sets
    """

    x,y = edge
    cand_nodes = [z for z in G if z != x]
    cand_dict = dict()
    for z in cand_nodes:
        paths = nx.all_simple_paths(G, source=x, target=z)
        for path in paths:
            for iter in range(len(path)-2):
                a,b,c = path[iter:iter+3]
                if not G.check_types(a, '*->' , b, '<-*', c) and not G.has_edge(a, b):
                    break
            else:
                cand_dict[z] = path
                break
    return cand_dict
    
