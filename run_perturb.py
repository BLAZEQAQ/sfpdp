import os.path

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import stats

from perturbation import perturb_laplace

from data_loader import load_data


def degreeDistribution_KSDistance(A, A_dp):

    degreeList = np.sum(A, axis=1)
    degreeSort = np.sort(degreeList)
    cdf = np.arange(1, len(degreeSort)+1)/len(degreeSort)

    degreeList_dp = np.sum(A_dp, axis=1)
    degreeSort_dp = np.sort(degreeList_dp)
    cdf_dp = np.arange(1, len(degreeSort_dp) + 1) / len(degreeSort_dp)

    distance, pvalue = stats.ks_2samp(degreeList, degreeList_dp)
    return distance


def cosine_similarity(attr_matrix1, attr_matrix2):
    attr_matrix1 = attr_matrix1.T
    attr_matrix2 = attr_matrix2.T
    # 计算两个向量的点积
    dot_product = np.sum(attr_matrix1 * attr_matrix2, axis=1)

    # 计算两个向量的范数
    norm1 = np.linalg.norm(attr_matrix1, axis=1)
    norm2 = np.linalg.norm(attr_matrix2, axis=1)

    # 避免除以零
    norm1 = np.where(norm1 == 0, 1, norm1)
    norm2 = np.where(norm2 == 0, 1, norm2)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm1 * norm2)

    # 取平均余弦相似度作为两个图的相似度
    average_cosine_sim = np.mean(cosine_sim)

    return average_cosine_sim



def compute_wl_hash(A, X, iteration=1):
    n = A.shape[0]
    H = X.copy()  # 初始化哈希矩阵为属性矩阵的副本

    for _ in range(iteration):
        H_new = np.empty(n, dtype=object)
        for i in range(n):
            # 获取邻居节点的属性索引
            neighbors = np.nonzero(A[i,:])[0]
            # 获取邻居节点的属性，确保是一维数组
            neighbor_attrs = np.array([H[j] for j in neighbors]).ravel()
            # 合并当前节点属性和邻居节点属性，并排序
            combined_attrs = np.sort(np.concatenate((H[i], neighbor_attrs)))
            H_new[i] = tuple(combined_attrs)
        H = H_new

    return H

def weisfeiler_lehman_similarity(A1, X1, A2, X2):
    H1 = compute_wl_hash(A1, X1)
    H2 = compute_wl_hash(A2, X2)

    # 计算哈希值的匹配数量
    match_count = sum(H1[i] == H2[i] for i in range(len(H1)))
    # 计算相似度
    similarity = match_count / len(H1)

    return similarity



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



    utility_A_list_SFPDP.append(np.sum(np.sum(abs(A_dp_SFPDP - A))))
    utility_W_list_SFPDP.append(np.sum(np.sum(abs(W_dp_SFPDP - W))))
    utility_A_list_SFPDP_MMI.append(np.sum(np.sum(abs(A_dp_SFPDP_MMI - A))))
    utility_W_list_SFPDP_MMI.append(np.sum(np.sum(abs(W_dp_SFPDP_MMI - W))))
    utility_A_list_SFPDP_LMLT.append(np.sum(np.sum(abs(A_dp_SFPDP_LMLT - A))))
    utility_W_list_SFPDP_LMLT.append(np.sum(np.sum(abs(W_dp_SFPDP_LMLT - W))))

    ks_ditance_list_SFPDP.append(degreeDistribution_KSDistance(A, A_dp_SFPDP))
    ks_ditance_list_SFPDP_MMI.append(degreeDistribution_KSDistance(A, A_dp_SFPDP_MMI))
    ks_ditance_list_SFPDP_LMLT.append(degreeDistribution_KSDistance(A, A_dp_SFPDP_LMLT))

    cosine_ditance_list_SFPDP.append(cosine_similarity(W, W_dp_SFPDP))
    cosine_ditance_list_SFPDP_MMI.append(cosine_similarity(W, W_dp_SFPDP_MMI))
    cosine_ditance_list_SFPDP_LMLT.append(cosine_similarity(W, W_dp_SFPDP_LMLT))

    wl_kernel_list_SFPDP.append(weisfeiler_lehman_similarity(A, W, A_dp_SFPDP, W_dp_SFPDP))
    wl_kernel_list_SFPDP_MMI.append(weisfeiler_lehman_similarity(A, W, A_dp_SFPDP_MMI, W_dp_SFPDP_MMI))
    wl_kernel_list_SFPDP_LMLT.append(weisfeiler_lehman_similarity(A, W, A_dp_SFPDP_LMLT, W_dp_SFPDP_LMLT))

print('===========================print metric=================================')
print('Utility for adjacency matrix:')
print(utility_A_list_SFPDP)
print(utility_A_list_SFPDP_MMI)
print(utility_A_list_SFPDP_LMLT)
print('Utility for attribute matrix:')
print(utility_W_list_SFPDP)
print(utility_W_list_SFPDP_MMI)
print(utility_W_list_SFPDP_LMLT)
print('KS distance for degree:')
print(ks_ditance_list_SFPDP)
print(ks_ditance_list_SFPDP_MMI)
print(ks_ditance_list_SFPDP_LMLT)
print('Cosine distance for attribute:')
print(cosine_ditance_list_SFPDP)
print(cosine_ditance_list_SFPDP_MMI)
print(cosine_ditance_list_SFPDP_LMLT)
print('WL kernel for graph:')
print(wl_kernel_list_SFPDP)
print(wl_kernel_list_SFPDP_MMI)
print(wl_kernel_list_SFPDP_LMLT)

print('===========================save figs=================================')
fig_save_path = './figs/test_figs/{}'.format(name)
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# Utility for adjacency matrix
plt.close('all')
plt.plot(epsilons, utility_A_list_SFPDP, label='SFP-DP', color='blue', marker='+', linewidth=2)
plt.plot(epsilons, utility_A_list_SFPDP_MMI, label='SFP-DP-MMI', color='red', marker='+', linewidth=2)
plt.plot(epsilons, utility_A_list_SFPDP_LMLT, label='SFP-DP-LMLT', color='orange', marker='+', linewidth=2)
plt.legend()
plt.title('variance for adjacency matrix')
plt.xlabel('privacy budget')
plt.ylabel('variance')
plt.savefig(os.path.join(fig_save_path,'variance_adjacency_{}.png'.format(name.lower())), dpi=1000, bbox_inches='tight')

# Utility for attribute matrix
plt.close('all')
plt.plot(epsilons, utility_W_list_SFPDP, label='SFP-DP', color='blue', marker='+', linewidth=2)
plt.plot(epsilons, utility_W_list_SFPDP_MMI, label='SFP-DP-MMI', color='red', marker='+', linewidth=2)
plt.plot(epsilons, utility_W_list_SFPDP_LMLT, label='SFP-DP-LMLT', color='orange', marker='+', linewidth=2)
plt.legend()
plt.title('variance for attribute matrix')
plt.xlabel('privacy budget')
plt.ylabel('variance')
plt.savefig(os.path.join(fig_save_path,'variance_attribute_{}.png'.format(name.lower())), dpi=1000, bbox_inches='tight')

# KS distance for degree
plt.close('all')
plt.plot(epsilons, ks_ditance_list_SFPDP, label='SFP-DP', color='blue', marker='+', linewidth=2)
plt.plot(epsilons, ks_ditance_list_SFPDP_MMI, label='SFP-DP-MMI', color='red', marker='+', linewidth=2)
plt.plot(epsilons, ks_ditance_list_SFPDP_LMLT, label='SFP-DP-LMLT', color='orange', marker='+', linewidth=2)
plt.legend()
plt.title('ks distance for degree')
plt.xlabel('privacy budget')
plt.ylabel('ks distance')
plt.savefig(os.path.join(fig_save_path,'ks_distance_{}.png'.format(name.lower())), dpi=1000, bbox_inches='tight')

# Cosine distance for attribute
plt.close('all')
plt.plot(epsilons, cosine_ditance_list_SFPDP, label='SFP-DP', color='blue', marker='+', linewidth=2)
plt.plot(epsilons, cosine_ditance_list_SFPDP_MMI, label='SFP-DP-MMI', color='red', marker='+', linewidth=2)
plt.plot(epsilons, cosine_ditance_list_SFPDP_LMLT, label='SFP-DP-LMLT', color='orange', marker='+', linewidth=2)
plt.legend()
plt.title('cosine distance for attribute')
plt.xlabel('privacy budget')
plt.ylabel('cosine distance')
plt.savefig(os.path.join(fig_save_path,'cosine_distance_{}.png'.format(name.lower())), dpi=1000, bbox_inches='tight')

# WL kernel for graph
plt.close('all')
plt.plot(epsilons, wl_kernel_list_SFPDP, label='SFP-DP', color='blue', marker='+', linewidth=2)
plt.plot(epsilons, wl_kernel_list_SFPDP_MMI, label='SFP-DP-MMI', color='red', marker='+', linewidth=2)
plt.plot(epsilons, wl_kernel_list_SFPDP_LMLT, label='SFP-DP-LMLT', color='orange', marker='+', linewidth=2)
plt.legend()
plt.title('wl kernel for graph')
plt.xlabel('privacy budget')
plt.ylabel('wl kernel')
plt.savefig(os.path.join(fig_save_path,'wl_kernel_{}.png'.format(name.lower())), dpi=1000, bbox_inches='tight')

