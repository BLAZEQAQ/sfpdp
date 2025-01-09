import networkx as nx
import igraph
import numpy as np
import pandas as pd


def Terrorist():
    terrorist_content_pd = pd.read_csv('./Original/terrorist_attack.nodes', sep='\t', header=None)
    # print(terrorist_content_pd)
    content_idx = list(terrorist_content_pd.index)
    paper_id = list(terrorist_content_pd.iloc[:, 0])
    nodesIndex = dict(zip(paper_id, content_idx))
    indexNodes = dict(zip(content_idx, paper_id))

    feature = terrorist_content_pd.iloc[:, 1:]
    feature = np.array(feature)
    print(feature.shape)
    print(feature)

    labelsDict = {'http://counterterror.mindswap.org/2005/terrorism.owl#Arson': 'a',
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Bombing': 'b',
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Kidnapping': 'c',
                  'http://counterterror.mindswap.org/2005/terrorism.owl#NBCR_Attack': 'd',
                  'http://counterterror.mindswap.org/2005/terrorism.owl#other_attack': 'e',
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Weapon_Attack': 'f'}
    dictLabels = dict(zip(labelsDict.values(), labelsDict.keys()))
    labelsInt = {'http://counterterror.mindswap.org/2005/terrorism.owl#Arson': 1,
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Bombing': 2,
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Kidnapping': 3,
                  'http://counterterror.mindswap.org/2005/terrorism.owl#NBCR_Attack': 4,
                  'http://counterterror.mindswap.org/2005/terrorism.owl#other_attack': 5,
                  'http://counterterror.mindswap.org/2005/terrorism.owl#Weapon_Attack': 6}
    labels = np.array(terrorist_content_pd.iloc[:, -1])
    for i in range(labels.shape[0]):
        labels[i] = labelsDict[labels[i]]
    for i in range(feature.shape[0]):
        feature[i][feature.shape[1] - 1] = labelsInt[feature[i][feature.shape[1] - 1]]

    terrorist_cites_pd = pd.read_csv('./Original/terrorist_attack_loc.edges', sep='\t', header=None)
    # print(terrorist_cites_pd.shape)
    terrorist_cite = np.array(terrorist_cites_pd)
    terrorist_cites = []
    # print(type(terrorist_cite[0][0]))
    for i in range(terrorist_cite.shape[0]):
        terrorist_cites.append(np.array(terrorist_cite[i][0].split(' ')))
    # terrorist_cites[:, [0, 1]] = terrorist_cites[:, [1, 0]]
    terrorist_cites = np.array(terrorist_cites)
    # print(terrorist_cites.shape)
    # print(terrorist_cites[0])

    A = np.zeros((feature.shape[0], feature.shape[0]))
    G = nx.DiGraph()
    nodes = []
    for i in range(terrorist_cites.shape[0]):
        if terrorist_cites[i][0] not in nodes:
            nodes.append(terrorist_cites[i][0])
        if terrorist_cites[i][1] not in nodes:
            nodes.append(terrorist_cites[i][1])
    for node in nodesIndex.keys():
        if node not in nodes:
            nodes.append(node)
    G.add_nodes_from(nodes)
    for i in range(terrorist_cites.shape[0]):
        G.add_edge(terrorist_cites[i][0], terrorist_cites[i][1])
        x = nodesIndex[terrorist_cites[i][0]]
        y = nodesIndex[terrorist_cites[i][1]]
        A[x][y] = 1
    # print(A.shape)
    # print(len(G.nodes))
    # print(A)
    # print(np.sum(np.sum(A)))
    # print(feature[:, -1])

    D = np.diag(np.sum(A, axis=1))
    L = D - A
    U, Sigular, V = np.linalg.svd(L)
    Sigular = np.diag(Sigular)

    np.savetxt('./arrayForm/adjacency_terrorist.txt', A, fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/feature_terrorist.txt', feature[:, 0:-1].astype(int), fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/label_terrorist.txt', feature[:, -1].astype(int), fmt='%d', delimiter=' ')
    np.savetxt('./arrayForm/leftSigular_terrorist.txt', U, fmt='%.64f', delimiter=' ')
    np.savetxt('./arrayForm/rightSigular_terrorist.txt', V, fmt='%.64f', delimiter=' ')
    np.savetxt('./arrayForm/sigular_terrorist.txt', Sigular, fmt='%.64f', delimiter=' ')
    results = {'G':G, 'adjacency': A, 'feature': feature[:, 0:-1], 'labels': labels, 'labelsDict': labelsDict, 'dictLabels': dictLabels,
                'labelsInt': labelsInt, 'nodesIndex': nodesIndex, 'indexNodes': indexNodes}
    return results

# test
# results = Terrorist()
# print(results)
# print(results['feature'].shape)
# print(results['feature'])
