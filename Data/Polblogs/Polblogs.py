import networkx as nx
import igraph
import numpy as np


def Polblogs():
    data = './Original/polblogs.gml'
    G = igraph.Graph.Read_GML(data)

    nodes = list(G.vs['label'])
    nodesIndex = dict(zip(nodes, list(range(len(nodes)))))
    indexNodes = dict(zip(nodesIndex.values(), nodesIndex.keys()))
    labelsDict = {'left or liberal': 'a', 'right or conservative': 'b'}
    labelsInt = {'left or liberal': 0, 'right or conservative': 1}
    dictLabels = dict(zip(labelsDict.values(), labelsDict.keys()))
    labels = np.array([int(i) for i in G.vs['value']])
    types = []
    for item in G.vs['source']:
        type = set(item.split(","))
        for t in type:
            if t not in types:
                types.append(t)
    # print(igraph.Graph.is_directed(G))
    A = np.zeros((len(nodes), len(nodes)))
    for edge in G.es:
        source = G.vs[edge.source]['label']
        target = G.vs[edge.target]['label']
        if source != target:
            A[nodesIndex[source]][nodesIndex[target]] = 1
    # print(A.shape)
    # print(A)

    features = np.zeros((len(nodes), len(types)))
    sources = list(G.vs['source'])
    for i in range(len(sources)):
        item = set(sources[i].split(","))
        for t in item:
            features[i][types.index(t)] = 1
    # print(features.shape)
    # print(features)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == 1:
                g.add_edge(nodes[i], nodes[j])
    # n = A.shape[0]
    # m = features.shape[1]
    features = features.astype(int)

    D = np.diag(np.sum(A, axis=1))
    L = D - A
    U, Sigular, V = np.linalg.svd(L)
    Sigular = np.diag(Sigular)

    np.savetxt('./arrayForm/adjacency_polblogs.txt', A, fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/feature_polblogs.txt', features, fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/label_polblogs.txt', labels, fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/leftSigular_polblogs.txt', U, fmt='%.64f', delimiter=' ')
    np.savetxt('./arrayForm/rightSigular_polblogs.txt', V, fmt='%.64f', delimiter=' ')
    np.savetxt('./arrayForm/sigular_polblogs.txt', Sigular, fmt='%.64f', delimiter=' ')
    results = {'G': g, 'adjacency': A, 'feature': features, 'labels': labels, 'labelsDict': labelsDict,
               'dictLabels': dictLabels,
               'labelsInt': labelsInt, 'nodesIndex': nodesIndex, 'indexNodes': indexNodes}
    return results


# # test
# results = Polblogs()
# print(results)
# print(results['feature'].shape)
# print(results['feature'])



