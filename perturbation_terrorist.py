from datetime import datetime
import numpy as np
# import cupy as cp

from data_loader import load_data
from weight_loader import load_weight


def generateFeature(A_dp, W_L):
    D_dp = np.diag(np.sum(A_dp , axis=1))
    L_dp = D_dp - A_dp
    U_dp, Sigular_dp, V_dp = np.linalg.svd(L_dp)
    W_noised_preliminary = np.dot(U_dp, W_L)
    # print('W_MI: ',np.max(W_noised_prime), np.min(W_noised_prime))
    # print(W_noised_prime)

    # W_dp = np.array(W_noised_prime > 1/2).astype(int)
    W_dp = np.array(W_noised_preliminary)
    return W_dp


def perturb_laplace(p, epsilon, K, Times, name, method):
    data = load_data(name)
    weights = load_weight(name, p)

    A = data['adjacency']
    W = data['feature']
    W_Laplace = data['feature_Laplace']

    n = W.shape[0]
    m = W.shape[1]

    sortIndex_array = np.loadtxt('./Weights/{}/sortIndex.txt'.format(name), dtype=int, delimiter=' ')

    C_dict = {'SFPDP_polblogs': (1e+18) * 1.2, 'SFPDP_terrorist': (1e+33)*1.56, 'SFPDP_cora': (1e+32) * 9.854,
              'SFPDP_MMI_polblogs': (1e+4) * 2.6, 'SFPDP_MMI_terrorist': (1e+18) * 2.46, 'SFPDP_MMI_cora': (1e+18) * 4.2,
              'SFPDP_LMLT_polblogs': (1e+15) * 2.87, 'SFPDP_LMLT_terrorist': (1e+18) * 9.454, 'SFPDP_LMLT_cora': (1e+18) * 4.2,}

    # C_dict = {'SFPDP_polblogs': (1e+12) * 4.2, 'SFPDP_terrorist': (1e+18) * 2.94, 'SFPDP_cora': (1e+32) * 9.854,
    #           'SFPDP_MMI_polblogs': (1e+2) * 4, 'SFPDP_MMI_terrorist': (1e+18) * 7.7, 'SFPDP_MMI_cora': (1e+18) * 4.2,
    #           'SFPDP_LMLT_polblogs': (1e+11) * 5.67, 'SFPDP_LMLT_terrorist': (1e+32) * 9.854,
    #           'SFPDP_LMLT_cora': (1e+18) * 4.2, }

    A_preliminary = np.zeros((n, n))


    if method == 'SFPDP':
        C_SFPDP = C_dict['{}_{}'.format(method, name.lower())]
        weight_SFPDP = weights['SFPDP']

        scales_SFPDP = np.sum(np.sum(weight_SFPDP)) / epsilon
        scales_SFPDP = scales_SFPDP / C_SFPDP

        noise_sample_SFPDP = np.zeros_like(A)
        for s in range(n):
            for t in range(n):
                if int(sortIndex_array[s][t]) - 1 in list(range(K)):

                    noise_sample_SFPDP[s][t] = np.mean(np.array(np.random.laplace(0, scales_SFPDP, Times)))
                    A_preliminary[s][t] = A[s][t] + noise_sample_SFPDP[s][t]
                else:
                    A_preliminary[s][t] = A[s][t]
        print('Noise ---- noises_SFPDP with max={} and min={}'.format(np.max(noise_sample_SFPDP), np.min(noise_sample_SFPDP)) )

    if method == 'SFPDP_MMI':
        C_SFPDP_MMI = C_dict['{}_{}'.format(method, name.lower())]
        weight_SFPDP_MMI = weights['SFPDP_MMI']

        weightSum_SFPDP_MMI = np.sum(np.sum(weight_SFPDP_MMI))

        scales_SFPDP_MMI = np.zeros((n, n))
        for s in range(n):
            for t in range(n):
                if s != t and sortIndex_array[s][t] != 0:
                    q = sortIndex_array[s][t]
                    C_q = np.power(np.sqrt(1 - p), 1 - q)
                    scales_SFPDP_MMI[s][t] = weightSum_SFPDP_MMI * C_q / epsilon

        scales_SFPDP_MMI_mask = np.isinf(scales_SFPDP_MMI)
        scales_SFPDP_MMI = np.where(scales_SFPDP_MMI_mask, 0, scales_SFPDP_MMI)
        scales_SFPDP_MMI = scales_SFPDP_MMI/C_SFPDP_MMI

        noise_sample_SFPDP_MMI = np.zeros_like(A)
        for s in range(n):
            for t in range(n):
                if scales_SFPDP_MMI[s][t] != 0 and int(sortIndex_array[s][t]) - 1 in list(range(K)):
                    noise_sample_SFPDP_MMI[s][t] = np.mean(np.array(np.random.laplace(0, scales_SFPDP_MMI[s][t], Times)))
                    A_preliminary[s][t] = A[s][t] + noise_sample_SFPDP_MMI[s][t]
                else:
                    A_preliminary[s][t] = A[s][t]
        print('Noise ---- noises_SFPDP_MMI with max={} and min={}'.format(np.max(noise_sample_SFPDP_MMI),
                                                                      np.min(noise_sample_SFPDP_MMI)))

    if method == 'SFPDP_LMLT':
        C_SFPDP_LMLT = C_dict['{}_{}'.format(method, name.lower())]
        weight_SFPDP_LMLT = weights['SFPDP_LMLT']

        weightSum_SFPDP_LMLT = np.sum(np.sum(weight_SFPDP_LMLT))

        scales_SFPDP_LMLT = np.zeros((n, n))
        for s in range(n):
            for t in range(n):
                if s != t and sortIndex_array[s][t] != 0:
                    q = sortIndex_array[s][t]
                    C_q = np.power(np.sqrt(1 - p), 1 - q)
                    scales_SFPDP_LMLT[s][t] = weightSum_SFPDP_LMLT * C_q / epsilon

        scales_SFPDP_LMLT_mask = np.isinf(scales_SFPDP_LMLT)
        scales_SFPDP_LMLT = np.where(scales_SFPDP_LMLT_mask, 0, scales_SFPDP_LMLT)
        scales_SFPDP_LMLT = scales_SFPDP_LMLT / C_SFPDP_LMLT

        noise_sample_SFPDP_LMLT = np.zeros_like(A)
        for s in range(n):
            for t in range(n):
                if scales_SFPDP_LMLT[s][t] != 0 and int(sortIndex_array[s][t]) - 1 in list(range(K)):
                    noise_sample_SFPDP_LMLT[s][t] = np.mean(np.array(np.random.laplace(0, scales_SFPDP_LMLT[s][t], Times)))
                    A_preliminary[s][t] = A[s][t] + noise_sample_SFPDP_LMLT[s][t]
                else:
                    A_preliminary[s][t] = A[s][t]

        print('Noise ---- noises_SFPDP_LMLT with max={} and min={}'.format(np.max(noise_sample_SFPDP_LMLT),
                                                                          np.min(noise_sample_SFPDP_LMLT)))


    np.fill_diagonal(A_preliminary, 0)
    A_dp = (A_preliminary > 1 / 2).astype(int)
    W_preliminary = generateFeature(A_dp, W_Laplace)

    print('before discretization: ')
    print('adjaency: max={}, min={} and diff_sum={}'.format(np.max(A_preliminary), np.min(A_preliminary), np.sum(np.sum(abs(A_preliminary - A)))))
    print('feature: max={}, min={} and diff_sum={}'.format(np.max(W_preliminary), np.min(W_preliminary), np.sum(np.sum(abs(W_preliminary - W)))))

    W_dp = (W_preliminary > 1/2).astype(int)

    print('after discretization: ')
    print('adjaency: max={}, min={} and diff_sum={}'.format(np.max(A_dp), np.min(A_dp), np.sum(np.sum(abs(A_dp - A)))))
    print('feature: max={}, min={} and diff_sum={}'.format(np.max(W_dp), np.min(W_dp), np.sum(np.sum(abs(W_dp - W)))))

    return A_dp, W_dp



