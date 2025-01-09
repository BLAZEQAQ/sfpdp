import networkx as nx
import igraph
import numpy as np


def load_data(name):
    adjacency = np.loadtxt('./Data/{}/arrayForm/adjacency_{}.txt'.format(name, name.lower()), dtype=int)
    feature = np.loadtxt('./Data/{}/arrayForm/feature_{}.txt'.format(name, name.lower()), dtype=int)
    label = np.loadtxt('./Data/{}/arrayForm/label_{}.txt'.format(name, name.lower()), dtype=int)
    U = np.loadtxt('./Data/{}/arrayForm/leftSigular_{}.txt'.format(name, name.lower()), dtype=float)
    V = np.loadtxt('./Data/{}/arrayForm/rightSigular_{}.txt'.format(name, name.lower()), dtype=float)
    Sigular = np.loadtxt('./Data/{}/arrayForm/sigular_{}.txt'.format(name, name.lower()), dtype=float)

    W_Laplace = np.dot(U.T, feature)

    data = {'adjacency': adjacency, 'feature': feature, 'label': label, 'leftSigular': U,
            'rightSigular': V, 'sigular': Sigular, 'feature_Laplace': W_Laplace}

    return data


# # test
# adjacency, feature, label = load_data('Polblogs')
# print(adjacency.shape)
# print(feature.shape)
# print(label.shape)









