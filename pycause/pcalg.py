"""
Implementation of the PC algorithm.
[1] Spirtes, Glymour, and Scheines, Causation, Prediction, and Search, 2nd ed.
[2] Kalisch & Buhlmann, J Machine Learning Res 8, 613 (2007)
"""

from numpy import shape
import skeleton as skel
import dirgraph
from septests import get_adjs

def pc(statdat,sig,verbose=True):
    """Run PC algorithm on statistical data.
    
    :param statdat: DataStats object containing input data
    :param sig: confidence interval for conditional independence test, expressed in units of standard deviation
    :returns:
        G: causal DAG
        sep_set: d-separation dict, key = separated node pairs, value = d-separating set
    """
    
    if statdat.var_names is not None:
        nodes = statdat.var_names.keys()
    else:
        nvars = shape(statdat.raw_data)[1]
        nodes = range(nvars)
    G = skel.init_skel(nodes) # initialize graph, including attributes
    G, sep_set = skel.get_skel(G,get_adjs,statdat,sig) # step B, p. 84 of [1]
    if verbose:
        print('Skeleton computed')
    G = dirgraph.DirGraph(G.to_directed())
    G = orient_colliders(G,sep_set) # step C, p. 84 of [1]
    G = orient_edges(G) # step D (not the same as p. 84 of [1], but equivalent formulation)
    return G, sep_set
    
def orient_colliders(skel, sep_set):
    """Identify and orient colliders in skeleton.
    
    :param skel: skeleton graph calculated by get_skel
    :return: partially oriented graph as determined by colliders
    """
    
    # only give definite orientation if collider can be uniquely oriented
    for edge in skel.edges():
        skel[edge[0]][edge[1]]['arrHead'] = False
    triples = [(x,y,z) for y in skel for x in skel.predecessors(y)
                        for z in skel.successors(y) if x<z]
    for (x,y,z) in triples:
        if (x,z) in sep_set.keys():
            this_sepset = sep_set[(x, z)]
        else:
            this_sepset = set([])
        if y not in this_sepset and skel.is_undir_edge((x, y)) and skel.is_undir_edge((y, z)):
            skel[x][y]['arrHead'] = True
            skel[z][y]['arrHead'] = True
    for edge in skel.edges():
        x,y = edge
        if skel.is_undir_edge(edge) and skel[x][y]['arrHead'] and not skel[y][x]['arrHead']:
            skel.remove_edge(*edge[::-1])
    return skel

def orient_edges(G):
    """Orient remaining edges after colliders have been oriented.
    
    :param G: partially oriented graph (colliders oriented)
    :returns: maximally oriented DAG
    """

    undir_list = [edge for edge in G.edges() if G.is_undir_edge(edge)]
    undir_len = len(undir_list)
    idx = 0
    while idx < undir_len:
        success = False
        for edge in undir_list:
            if can_orient(G,edge):
                G.remove_edge(*edge[::-1])
                success = True
        if success:
            undir_list = [edge for edge in G.edges() if G.is_undir_edge(edge)]
            idx += 1
        else:
            break
    return G

def can_orient(G,edge):
    """Test whether edge should be oriented under PC algorithm rules for non-collider edges.
    
    :param G: partially oriented graph
    :param edge: edge of G under test
    :returns: True if edge should be oriented, False otherwise
    """

    if not G.is_undir_edge(edge): # test that edge is not already oriented
        return False
    # tests from [2]
    x,y = edge
    testR1 = [z for z in G if G.is_dir_edge((z,x)) and G.is_non_adj((z, y))]
    testR2 = [z for z in G if G.is_dir_edge((x,z)) and G.is_dir_edge((z,y))]
    non_adj_list = [(z1,z2) for z1 in G for z2 in G if G.is_non_adj((z1, z2))]
    testR3 = [(z1,z2) for (z1,z2) in non_adj_list
              if G.is_undir_edge((x, z1)) and G.is_dir_edge((z1, y))
              and G.is_undir_edge((x, z2)) and G.is_dir_edge((z2, y))]
    testR4 = [(z1,z2) for (z1,z2) in non_adj_list
              if G.is_undir_edge((x, z1)) and G.is_dir_edge((z1, z2)) and G.is_dir_edge((z2, y))]
    if testR1 or testR2 or testR3 or testR4:
        return True
    else:
        return False
