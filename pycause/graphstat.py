"""
Generate statistical data for a graph using a linear Gaussian model.
"""

from copy import deepcopy
import numpy as np

## NOTE this version randomizes betas on each trial!

def lingauss(graph_in, beta_range, stdev, fname, nsamp=1000):
    """
    Generate statistical data for a graph using a linear Gaussian model.
    Linear coefficients are randomly chosen in a range specified by beta_range.
    TODO 2: replace vals with generator?
    :param graph_in: graph to be modeled
    :param beta_range: range [beta_min, beta_max] for linear coefficients
    :param stdev: standard deviation of Gaussian noise added at each node
    :param fname: filename for saving data
    :param nsamp: number of data samples to generate
    :return: output = numpy array of data samples
    """

    # decouple side effects from input graph
    graph = deepcopy(graph_in)    
    # apply standard nomenclature to all nodes
    for node in graph.nodes():
        graph.node[node]['name'] = 'var_' + str(node)
    
    # calculate data values
    node_names = [graph.node[node]['name'] for node in graph]
    vals = dict(zip(node_names, [np.zeros(nsamp) for _ in range(len(node_names))]))
    for trial in range(nsamp):
        iter = 0
        calc_names = list()
        # generate values for nodes, one layer at a time
        while set(calc_names) < set(node_names) and iter < len(graph):
            vals, new_names = get_lg_vals(graph, calc_names, trial, vals, beta_range, stdev)
            calc_names += new_names
            iter += 1

    # generate and save traces (= statistical realization of variables)
    output = np.column_stack([vals[node_name][:] for node_name in node_names])
    np.savetxt(fname,output,fmt='%.4f',delimiter='\t')
    return output

def get_lg_vals(graph, calc_names, trial, vals, beta_range, stdev):
    """
    Generate values of nodes for next layer of graph, given values of previous layers.
    :param graph: graph to be modeled
    :param calc_names: names of all nodes that have already been assigned values
    :param trial: number of current trial
    :param vals: values of all data in all trials
    :param beta_range: range [beta_min, beta_max] for linear coefficients
    :param stdev: standard deviation of Gaussian noise added at each node
    :return: vals = values of nodes at next layer
             new_names = names of nodes at next layer
    """

    old_nodes = [node for node in graph if graph.node[node]['name'] in calc_names]
    new_nodes = [node for node in graph if node not in old_nodes and set(graph.predecessors(node)) <= set(old_nodes)]
    new_names = list()
    for node in new_nodes:
        node_name = graph.node[node]['name']
        parent_names = [graph.node[parent]['name'] for parent in graph.predecessors(node)]
        parent_vals = [vals[parent_name][trial] for parent_name in parent_names]
        if parent_vals:
            betas = np.array([np.random.uniform(*beta_range) for _ in range(len(parent_vals))])
            betas = betas / sum(betas)
            for iter in range(len(betas)):
                vals[node_name][trial] += betas[iter] * parent_vals[iter]
            vals[node_name][trial] += np.random.normal(0, max(stdev, 1e-9))
        else:
            vals[node_name][trial] = np.random.normal()
        new_names.append(node_name)
    return vals, new_names
