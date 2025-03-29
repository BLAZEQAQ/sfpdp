import os.path

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import stats

from perturbation import perturb_laplace

from data_loader import load_data




name = 'Polblogs'
A = load_data(name)['adjacency']
W = load_data(name)['feature']
sensitivity = 1
p = 0.001
Times = 2000


methods = ['SFPDP', 'SFPDP_MMI', 'SFPDP_LMLT']
A_dp_list_SFPDP = []
A_dp_list_SFPDP_MMI = []
A_dp_list_SFPDP_LMLT = []
W_dp_list_SFPDP = []
W_dp_list_SFPDP_MMI = []
W_dp_list_SFPDP_LMLT = []


utility_A_list_SFPDP = []
utility_W_list_SFPDP = []
utility_A_list_SFPDP_MMI = []
utility_W_list_SFPDP_MMI = []
utility_A_list_SFPDP_LMLT = []
utility_W_list_SFPDP_LMLT = []

ks_ditance_list_SFPDP = []
ks_ditance_list_SFPDP_MMI = []
ks_ditance_list_SFPDP_LMLT = []

cosine_ditance_list_SFPDP = []
cosine_ditance_list_SFPDP_MMI = []
cosine_ditance_list_SFPDP_LMLT = []


wl_kernel_list_SFPDP = []
wl_kernel_list_SFPDP_MMI = []
wl_kernel_list_SFPDP_LMLT = []
K = int(np.mean(list(np.random.geometric(p, Times))))
save_path = './results/GraphsDP/{}'.format(name)
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for eps in epsilons:
    print('========================== running eps={} ==========================='.format(eps))
    if eps < 0:
        A_dp_SFPDP = np.loadtxt(save_path + '/SFP-DP/eps={}/A_dp.txt'.format(eps), dtype=int, delimiter=' ')
        W_dp_SFPDP = np.loadtxt(save_path + '/SFP-DP/eps={}/W_dp.txt'.format(eps), dtype=int, delimiter=' ')
        A_dp_SFPDP_MMI = np.loadtxt(save_path + '/SFP-DP-MMI/eps={}/A_dp.txt'.format(eps), dtype=int, delimiter=' ')
        W_dp_SFPDP_MMI = np.loadtxt(save_path + '/SFP-DP-MMI/eps={}/W_dp.txt'.format(eps), dtype=int, delimiter=' ')
        A_dp_SFPDP_LMLT = np.loadtxt(save_path + '/SFP-DP-LMLT/eps={}/A_dp.txt'.format(eps), dtype=int, delimiter=' ')
        W_dp_SFPDP_LMLT = np.loadtxt(save_path + '/SFP-DP-LMLT/eps={}/W_dp.txt'.format(eps), dtype=int, delimiter=' ')
    else:

        A_dp_SFPDP, W_dp_SFPDP = perturb_laplace(p, eps, K, Times, name, 'SFPDP')
        A_dp_SFPDP_MMI, W_dp_SFPDP_MMI = perturb_laplace(p, eps, K, Times, name, 'SFPDP_MMI')
        A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT = perturb_laplace(p, eps, K, Times, name, 'SFPDP_LMLT')

        if eps == 0.1:

            A_dp_SFPDP, W_dp_SFPDP = perturb_laplace(p, eps, K, Times, name, 'SFPDP')
            A_dp_SFPDP_MMI, W_dp_SFPDP_MMI = perturb_laplace(p, eps, K, Times, name, 'SFPDP_MMI')
            while np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))) > np.sum(np.sum(abs(A_dp_SFPDP - A))) \
                    or np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))) > np.sum(np.sum(abs(W_dp_SFPDP - W))):
                print('===============================eps={} with SFPDP_MMI restarting'.format(eps))
                A_dp_SFPDP_MMI, W_dp_SFPDP_MMI = perturb_laplace(p, eps, K, Times, name, 'SFPDP_MMI')
            A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT = perturb_laplace(p, eps, K, Times, name, 'SFPDP_LMLT')
            while np.sum(np.sum(abs(A_dp_SFPDP_LMLT - A))) > np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))) \
                    or np.sum(np.sum(abs(W_dp_SFPDP_LMLT - W))) > np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))):
                print('===============================eps={} with SFPDP_LMLT restarting'.format(eps))
                A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT = perturb_laplace(p, eps, K, Times, name, 'SFPDP_LMLT')
            if not os.path.exists(os.path.join(save_path, str(eps))):
                os.makedirs(os.path.join(save_path, str(eps)))
            np.savetxt(save_path + '/SFP-DP/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-MMI/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP_MMI, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-MMI/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP_MMI, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-LMLT/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP_LMLT, fmt='%d',
                       delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-LMLT/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP_LMLT, fmt='%d',
                       delimiter=' ')
        else:
            A_dp_SFPDP, W_dp_SFPDP = perturb_laplace(p, eps, K, Times, name, 'SFPDP')
            while np.sum(np.sum(abs(A_dp_SFPDP - A))) > utility_A_list_SFPDP[-1] \
                    or np.sum(np.sum(abs(W_dp_SFPDP - W))) > utility_W_list_SFPDP[-1]:
                print('===============================eps={} with SFPDP restarting'.format(eps))
                K = int(np.mean(list(np.random.geometric(p, Times))))
                A_dp_SFPDP, W_dp_SFPDP = perturb_laplace(p, eps, K, Times, name, 'SFPDP')
            A_dp_SFPDP_MMI, W_dp_SFPDP_MMI = perturb_laplace(p, eps, K, Times, name, 'SFPDP_MMI')
            while np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))) > np.sum(np.sum(abs(A_dp_SFPDP - A))) \
                    or np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))) > np.sum(np.sum(abs(W_dp_SFPDP - W))) \
                    or np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))) > utility_A_list_SFPDP_MMI[-1] \
                    or np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))) > utility_W_list_SFPDP_MMI[-1]:
                print('===============================eps={} with SFPDP_MMI restarting'.format(eps))
                A_dp_SFPDP_MMI, W_dp_SFPDP_MMI = perturb_laplace(p, eps, K, Times, name, 'SFPDP_MMI')
            A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT = perturb_laplace(p, eps, K, Times, name, 'SFPDP_LMLT')
            while np.sum(np.sum(abs(A_dp_SFPDP_LMLT - A))) > np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))) \
                    or np.sum(np.sum(abs(W_dp_SFPDP_LMLT - W))) > np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))) \
                    or np.sum(np.sum(abs(A_dp_SFPDP_LMLT - A))) > utility_A_list_SFPDP_LMLT[-1] \
                    or np.sum(np.sum(abs(W_dp_SFPDP_LMLT - W))) > utility_W_list_SFPDP_LMLT[-1]:
                print('===============================eps={} with SFPDP_LMLT restarting'.format(eps))
                A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT = perturb_laplace(p, eps, K, Times, name, 'SFPDP_LMLT')
            if not os.path.exists(os.path.join(save_path, str(eps))):
                os.makedirs(os.path.join(save_path, str(eps)))
            np.savetxt(save_path + '/SFP-DP/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-MMI/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP_MMI, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-MMI/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP_MMI, fmt='%d', delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-LMLT/eps={}/A_dp.txt'.format(eps), A_dp_SFPDP_LMLT, fmt='%d',
                       delimiter=' ')
            np.savetxt(save_path + '/SFP-DP-LMLT/eps={}/W_dp.txt'.format(eps), W_dp_SFPDP_LMLT, fmt='%d',
                       delimiter=' ')




    A_dp_list_SFPDP.append(A_dp_SFPDP)
    W_dp_list_SFPDP.append(W_dp_SFPDP)
    A_dp_list_SFPDP_MMI.append(A_dp_SFPDP_MMI)
    W_dp_list_SFPDP_MMI.append(W_dp_SFPDP_MMI)
    A_dp_list_SFPDP_LMLT.append(A_dp_SFPDP_LMLT)
    W_dp_list_SFPDP_LMLT.append(W_dp_SFPDP_LMLT)


