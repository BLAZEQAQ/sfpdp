import numpy as np

# from Weights import weightPreliminary as Preliminary
# from Weights import weight_SFPDP as SFPDP
# from Weights import weight_SFPDP_MMI as SFPDP_MMI
# from Weights import weight_SFPDP_LMLT as SFPDP_LMLT

from weight_calculation import weightSortCalculation
from weight_calculation import sortIndex
from weight_calculation import caculateWeight_SFPDP
from weight_calculation import caculateWeight_SFPDP_MMI
from weight_calculation import caculateWeight_SFPDP_LMLT
import weight_calculation
from data_loader import load_data

name = 'Cora'
sensitivity = 1
p = 0.001

data = load_data(name)
A = data['adjacency']
W = data['feature']
U = data['leftSigular']
V = data['rightSigular']
Sigular = data['sigular']
W_Laplace = data['feature_Laplace']

left_pinv, right_pinv = weight_calculation.caculateWdight_outer(U, V, Sigular, name)
# # weightsSort = weightSortCalculation(U, V, W_Laplace, name)
weights_SFPDP_MMI = caculateWeight_SFPDP_MMI(U, V, W_Laplace, sensitivity, name)
sortIndex_array = sortIndex(name)
weights_SFPDP_LMLT = caculateWeight_SFPDP_LMLT(U, V, W_Laplace, sensitivity, name)
weights_SFPDP = caculateWeight_SFPDP(U, V, W_Laplace, p, sensitivity, name)



