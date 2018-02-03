"""
Operations on graph skeleton
"""

from itertools import combinations
import networkx as nx

def init_skel(nodes):
    """Initializes graph from node list in preparation for skeleton retrieval.
    
    :param nodes: list of node names
    :return: G = networkx.Graph object with 'dep_stat' attribute for each edge, used to hold statistical dependence
    test statistic.
    """

    G = nx.complete_graph(len(nodes))
    # each edge carries information on statistical dependence
    for edge in G.edges():
        G[edge[0]][edge[1]]['dep_stat'] = {}
    return G

def get_skel(G, get_poss_sep, statdat, sig):
    """Removes edges between conditionally independent nodes of completely connected graph.
    
    :param G: networkx.Graph object, normally completely connected
    :param get_poss_sep: function to retrieve possible conditioning sets. Depends on algorithm.
    :param statdat: datastats.DataStats object containing statistical data for independence tests
    :param sig: level of significance for independence tests, expressed in # standard deviations
    :return: G = original 'G' with conditionally independent edges removed. Note: not a copy of G, but the original!
             sepset = dictionary of d-separation sets. key: d-separated node pair, value: d-separating set for key.
    """

    card = 0
    card_max = G.number_of_nodes()
    num_edges = G.number_of_edges()
    sepset = dict()
    poss_seps = dict()
    for edge in G.edges():
        poss_seps[edge] = get_poss_sep(G,edge)
    # loop over cardinality of trial sepsets
    while card<card_max:   
        nedge = 0
        tested_edges = set()
        # loop over edges
        while nedge<num_edges:  
            if set(G.edges())<=set(tested_edges):
                break
            edge = [edge for edge in G.edges() if edge not in tested_edges][0]
            # test all trial sepsets of cardinality card
            G,edge_sepset = find_edge_sep(G, statdat, edge, poss_seps[edge], card, sig)
            if edge_sepset is not None:
                G.remove_edge(*edge)
                sepset[edge] = edge_sepset
            tested_edges.add(edge)
            nedge += 1
        card += 1
        if max([G.degree(x) for x in G])<=card:
            break
    return G,sepset


def find_edge_sep(G, statdat, edge, poss_seps, card, sig):
    """Looks for d-separation of a node pair by all conditioning sets of a given cardinality.
    
    :param G: networkx.Graph object
    :param statdat: datastats.DataStats object containing statistical data for independence tests
    :param edge: candidate node pair (x,y)
    :param poss_seps: set of all candidate nodes for inclusion in conditioning set
    :param card: cardinality of conditioning set
    :param sig: level of significance for independence tests, expressed in # standard deviations
    :return: G = original 'G' with statistical dependence information (dep_stat,stat_test) added to edge
                dep_stat = conditional dependence statistic (e.g., partial correlation)
                stat_test = test statistic for dep_stat (e.g., Fisher's Z)
             cond = conditioning set that successfully d-separates 'edge', = None if no such set exists
    """

    get_cond_dep = statdat.get_part_corr 
    x, y = edge
    for poss in combinations(poss_seps, card):
        cond = set(poss)
        cond_key = tuple(sorted(cond))
        # get conditional dependence statistic
        if cond_key not in G[x][y]['dep_stat'].keys():
            dep_stat,stat_test = get_cond_dep(edge, cond)
            # push information onto graph edge
            G[x][y]['dep_stat'][cond_key] = (dep_stat, stat_test)
        else:
            dep_stat, stat_test = G[x][y]['dep_stat'][cond_key]
        # if separating set is found, quit looking
        if abs(stat_test)<sig:
            return G, cond
    else: # no sets of cardinality 'card' separate the edge under test
        return G, None


def find_test_edge(G, tested_edges, card):
    """Find new eligible edge for test of d-separation with respect to conditioning sets of a given cardinality.
    
    :param G: networkx.Graph object
    :param tested_edges: list of edges already tested at this cardinality
    :param card: cardinality under test
    """
    
    untested = [edge for edge in G.edges() if edge not in tested_edges]
    for edge in untested:
        if len(G.neighbors(edge[0])) >= card + 1:
            return edge
    else: # no eligible edges found
        return None