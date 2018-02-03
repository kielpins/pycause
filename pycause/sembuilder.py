"""
Generate structural equation model for a graph.
"""

import numpy as np
import networkx as nx

def make_lgfun(betas, stdev):
    stdev = max(1e-9,stdev)
    def lgfun(vars):
        retval = 0
        for iter in range(len(betas)):
            retval += betas[iter]*vars[iter]
        retval += np.random.normal(0,stdev)
        return retval
    return lgfun

def test_parse():
    graph = nx.DiGraph([(0,2),(3,1),(2,1),(0,1),(1,4)])
    node_order,parents = parse(graph)
    assert node_order == [0,3,2,1,4]
    assert parents == {0:[],1:[0,2,3],2:[0],3:[],4:[1]}

def parse(graph):
    node_order = []
    parents = dict()
    iter = 0
    while set(node_order) < set(graph.nodes()) and iter < len(graph):
        next_tier = [node for node in graph if node not in node_order and 
                    set(graph.predecessors(node)) <= set(node_order)]
        for node in next_tier:
            parents[node] = sorted(graph.predecessors(node))
        node_order += sorted(next_tier)
        iter += 1
    return node_order,parents
        
def test_make_fun_list():
    parents = {0:[],1:[0]}
    beta = 0.5
    stdev = 0
    beta_range = (beta,beta)
    fun_list = make_fun_list(parents,beta_range,stdev)
    a = fun_list[0]([])
    b = fun_list[1]([a])
    assert np.isclose(beta*a,b)
    
def make_fun_list(parents,beta_range,stdev):
    fun_list = dict()
    for node in parents.keys():
        npar = len(parents[node])
        if npar == 0:
            sigma = 1
        else:
            sigma = stdev
        betas = np.random.uniform(*beta_range,size=npar)
        fun_list[node] = make_lgfun(betas,sigma)
    return fun_list

def test_do_trial():
    graph = nx.DiGraph([(0,2),(3,1),(2,1),(0,1),(1,4)])
    beta = 0.5
    node_order,parents = parse(graph)
    fun_list = make_fun_list(parents,(beta,beta),0)
    vals = do_trial(node_order,parents,fun_list)
    assert np.isclose(vals[2],beta*vals[0])
    assert np.isclose(vals[1],beta*(vals[3]+vals[2]+vals[0]))
    assert np.isclose(vals[4],beta*vals[1])
    
    graph = nx.DiGraph([(0,3),(1,3),(2,3),(3,4),(2,4)])
    beta = 0.5
    node_order,parents = parse(graph)
    fun_list = make_fun_list(parents,(beta,beta),0)
    vals = do_trial(node_order,parents,fun_list)
    assert np.isclose(vals[4],beta*(vals[2]+vals[3]))
    assert np.isclose(vals[3],beta*(vals[0]+vals[1]+vals[2]))

def do_trial(node_order,parents,fun_list):
    num = len(node_order)
    vals = np.zeros(num)
    for iter in range(num):
        node = node_order[iter]
        arglist = [vals[p] for p in parents[node]]
        vals[node] = fun_list[node](arglist)
    return vals

def sim_graph(graph,beta_range,stdev,fname,nsamp=1000):
    node_order,parents = parse(graph)
    fun_list = make_fun_list(parents,beta_range,stdev)
    data = np.zeros([nsamp,len(graph)])
    for iter in range(nsamp):
        data[iter,:] = do_trial(node_order,parents,fun_list)
    np.savetxt(fname,data,fmt='%.4f',delimiter='\t')
    return data
    
def test_sim_graph():
    graph = nx.DiGraph([(0,2),(3,1),(2,1),(0,1),(1,4)])
    beta = 0.5
    data = sim_graph(graph,(beta,beta),10)
    assert np.isclose(data[:,2],beta*data[:,0]).all()
    assert np.isclose(data[:,1],beta*(data[:,0]+data[:,2]+data[:,3])).all()
    assert np.isclose(data[:,4],beta*data[:,1]).all()
    
    
    