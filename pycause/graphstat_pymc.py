"""
Generates statistical data from graphical model using pymc implementation of Markov chain Monte Carlo.
"""

from copy import deepcopy
import numpy as np
import pymc

def get_lg_funs(graph, funs, beta_range, stdev):
    """Creates PyMC variables for all children of the variables specified in 'funs', using linear Gaussian structural
    equation model. Adds new variables to the 'funs' dictionary.
    
    Called while traversing directed acyclic graph for model-building.
    Uses structural equation model to generate data:
        child = beta1 * parent1 + beta2 * parent2 + ... + noise
        beta1, beta2, ... follow a uniform distribution over beta_range = (lower,upper)
    
    graph = networkx.DiGraph object specifying causal dependences
    funs = dict of PyMC variables already created. keys: node names, values: PyMC variables
    beta = linear coefficient in all structural equations
    stdev = standard deviation of exogenous noise added to each child variable
    
    Returns funs = dict of all extant PyMC variables, including children of variables in funs. keys: node names,
    values: PyMC variables
    """

    old_nodes = [node for node in graph if graph.node[node]['name'] in funs.keys()]
    new_nodes = [node for node in graph if node not in old_nodes and
                 set(graph.predecessors(node)) <= set(old_nodes)]
    for node in new_nodes:
        node_name = graph.node[node]['name']
        parent_names = [graph.node[parent]['name'] for parent in graph.predecessors(node)]
        parent_funs = [funs[parent_name] for parent_name in parent_names]
        if parent_funs:
            betas = np.array([np.random.uniform(*beta_range) for _ in range(len(parent_funs))])
            lc = pymc.LinearCombination(node_name, betas, parent_funs, 'doc')
            funs[node_name] = pymc.Normal(node_name, mu=lc, tau=1 / stdev ** 2)
        else:
            funs[node_name] = pymc.Normal(node_name, 0, 1)
    return funs


def lingauss(graph_in, beta_range, stdev, fname, nsamp=1000):
    """Generates linear Gaussian statistical data from directed acyclic graph.
    
    Uses structural equation model to generate data:
        child = beta1*parent1 + beta2*parent2 + ... + noise
        beta1,beta2,... follow a uniform distribution over beta_range = (lower,upper)
        noise is normally distributed with mean = 0 and standard deviation = stdev
    Saves data sampled from model in .txt format.
    
    graph_in = networkx DiGraph object specifying causal dependences
    beta_range = (lower,upper) bounds for linear coefficient in structural equations
    stdev = standard deviation of exogenous noise added to each child variable
    fname = filename for .txt save
    nsamp = number of data samples
    
    Returns mc = PyMC.MCMC object with nsamp samples.
    """

    # decouple side effects from input graph
    graph = deepcopy(graph_in)
    # apply standard nomenclature to all nodes
    for node in graph.nodes():
        graph.node[node]['name'] = 'var_' + str(node)

    # set up statistical functions
    node_names = [graph.node[node]['name'] for node in graph]
    funs = dict()
    iter = 0
    while set(funs.keys()) < set(node_names) and iter < len(graph):
        funs = get_lg_funs(graph, funs, beta_range, stdev)
        iter += 1

    # make PyMC model, generate and save traces (= statistical realization of variables)
    mod = pymc.Model([funs[node_name] for node_name in node_names])
    mc = pymc.MCMC(mod)
    samp = mc.sample(nsamp, progress_bar=False)
    output = np.column_stack([mc.trace(node_name)[:] for node_name in node_names])
    np.savetxt(fname, output, fmt='%.4f', delimiter='\t')
    return mc
