import concurrent
import logging
import os
from datetime import datetime
import numpy as np
import cupy as cp
import torch

from data_loader import load_data


# 前置的矩阵计算，与p无关
def caculateWdight_outer(U, V, Sigular, name):
    U_gpu = cp.array(U)
    V_gpu = cp.array(V)
    Sigular_gpu = cp.array(Sigular)


    left = cp.hstack((U_gpu.T, Sigular_gpu))
    # print('left: ', left.shape)
    left_pinv = cp.linalg.pinv(left)

    right = cp.vstack((Sigular_gpu, V_gpu))
    # print('right: ', right.shape)
    right_pinv = cp.linalg.pinv(right)

    cp.savetxt('./Weights/{}/weightsPreliminary_left.txt'.format(name), left_pinv, fmt='%.64f', delimiter=' ')
    cp.savetxt('./Weights/{}/weightsPreliminary_right.txt'.format(name), right_pinv, fmt='%.64f', delimiter=' ')

    # left_pinv = cp.loadtxt('./Weights/weight_AW/MI/weight_outer_left.txt', dtype=float, delimiter=' ')
    # right_pinv = cp.loadtxt('./Weights/weight_AW/MI/weight_outer_right.txt', dtype=float, delimiter=' ')

    return left_pinv, right_pinv


# 用于排序的权重计算，可以在运行MMI权重计算时一起完成，与p无关
def weightSortCalculation(U, V, W_L, name):
    # 将必要的数组转换为CuPy数组（假设它们适合放入GPU内存中）
    U_gpu = cp.array(U)
    V_gpu = cp.array(V)
    W_L_gpu = cp.array(W_L)

    n = W_L_gpu.shape[0]
    m = W_L_gpu.shape[1]

    weight_outer_left = cp.loadtxt('./Weights/{}/weightsPreliminary_left.txt'.format(name), dtype=float, delimiter=' ')
    weight_outer_right = cp.loadtxt('./Weights/{}/weightsPreliminary_right.txt'.format(name), dtype=float, delimiter=' ')

    weightsSort = cp.zeros((n, n))

    for s in range(n):
        begin = datetime.now()
        print(
            "sorted weights running Process {}, {}, {}, begin, info: {}".format(s, weight_outer_left.shape,
                                                                                        weight_outer_right.shape,
                                                                                        begin))
        for t in range(n):
            if s != t:
                I_st = cp.zeros((n, n))
                I_st[s][t] = -1
                I_st[s][s] = 1

                inner_left_list = cp.array([U_gpu[j].T @ I_st @ V_gpu[j] for j in range(n)])
                inner_left = cp.diag(inner_left_list)

                weight_inner = U_gpu.T @ I_st @ V_gpu - inner_left
                weight_st_pre = weight_outer_left @ weight_inner @ weight_outer_right
                weight_st = weight_st_pre[:n, :n]
                weight_st_w = cp.dot(weight_st, W_L_gpu)

                nonzero_mask = weight_st_w != 0
                weight_MI_pre_local = cp.zeros_like(weight_st_w)
                weight_MI_pre_local[nonzero_mask] = weight_st_w[nonzero_mask] ** 2

                weightsSort[s][t] = cp.mean(weight_MI_pre_local)




        end = datetime.now()
        run = end - begin
        print('Process {} run {} with sum={}, max={} and min={}:'.format(s, run, cp.sum(cp.sum(weightsSort)),
                                                                         cp.max(weightsSort),
                                                                         cp.min(weightsSort)))


    cp.savetxt('./Weights/{}/weightsSort.txt'.format(name), weightsSort, fmt='%.64f', delimiter=' ')

    # weight_MI = cp.loadtxt('./weight/weight_AW/MI/weight_MI.txt', dtype=float, delimiter=' ')

    return weightsSort


# 邻接矩阵元素排序，存储排序后每个元素的序列号，与p无关
def sortIndex(name):
    weightsSort = cp.loadtxt('./Weights/{}/weightsSort.txt'.format(name), dtype=float, delimiter=' ')
    n = weightsSort.shape[0]
    cp.fill_diagonal(weightsSort, 0)
    mask1 = cp.isinf(weightsSort)
    mask2 = cp.isnan(weightsSort)
    print(len(mask1), len(mask2))

    weightsSort_MMI = cp.where(mask1, 0, weightsSort)
    weightsSort_MMI = cp.where(mask2, 0, weightsSort)

    weightList = []
    indexList = []

    for s in range(n):
        print('sort add: ------------', s)
        for t in range(n):
            if weightsSort_MMI[s][t] > 0:
                weightList.append(weightsSort_MMI[s][t])
                indexList.append((s, t))
    print('sort: ')
    sortedlist = list(sorted(enumerate(weightList), key=lambda k: k[1], reverse=False))
    # sortIndex = []

    sortIndex_array = cp.zeros((n, n))

    for q in range(len(sortedlist)):
        index = indexList[sortedlist[q][0]]
        # sortIndex.append(index)
        sortIndex_array[index[0]][index[1]] = q + 1

    cp.savetxt('./Weights/{}/sortIndex.txt'.format(name), sortIndex_array, fmt='%d', delimiter=' ')
    # sortIndex_array = cp.loadtxt('./weight/weight_AW/MI/sortIndex_array.txt', dtype=int, delimiter=' ')
    print(cp.count_nonzero(sortIndex_array))
    return sortIndex_array


# 计算SFP-DP的权重，与p有关
def caculateWeight_SFPDP(U, V, W_L, p, sensitivity, name):
    # 将必要的数组转换为CuPy数组（假设它们适合放入GPU内存中）
    U_gpu = cp.array(U)
    V_gpu = cp.array(V)
    W_L_gpu = cp.array(W_L)

    n = W_L_gpu.shape[0]
    m = W_L_gpu.shape[1]

    weight_outer_left = cp.loadtxt('./Weights/{}/weightsPreliminary_left.txt'.format(name), dtype=float, delimiter=' ')
    weight_outer_right = cp.loadtxt('./Weights/{}/weightsPreliminary_right.txt'.format(name), dtype=float,
                                    delimiter=' ')
    sortIndex_array = cp.loadtxt('./Weights/{}/sortIndex.txt'.format(name), dtype=int, delimiter=' ')

    weight_preliminary = cp.zeros((n, m))
    for s in range(n):
        begin = datetime.now()

        print(
            "weights for SFP-DP running Process {}, {}, {}, begin, info: {}".format(s, weight_outer_left.shape,
                                                                                                     weight_outer_right.shape,
                                                                                                     begin))

        for t in range(n):
            if s != t:
                q = sortIndex_array[s][t]
                I_st = cp.zeros((n, n))
                I_st[s][t] = -1
                I_st[s][s] = 1

                inner_left_list = cp.array([U_gpu[j].T @ I_st @ V_gpu[j] for j in range(n)])
                inner_left = cp.diag(inner_left_list)

                weight_inner = U_gpu.T @ I_st @ V_gpu - inner_left
                weight_st_preliminary = weight_outer_left @ weight_inner @ weight_outer_right
                weight_st = weight_st_preliminary[:n, :n]
                weight_st_w = cp.dot(weight_st, W_L_gpu)

                nonzero_mask = weight_st_w != 0
                weight_preliminary_local = cp.zeros_like(weight_st_w)
                weight_preliminary_local[nonzero_mask] = weight_st_w[nonzero_mask] ** 2

                weight_preliminary_local = weight_preliminary_local * cp.power(1 - p, q - 1)
                # where_are_inf = cp.isinf(weight_preliminary_local)
                # weight_preliminary_local[where_are_inf] = 0
                nonzero_mask = weight_preliminary_local > 0
                weight_preliminary_s = cp.zeros_like(weight_preliminary)
                weight_preliminary_s[nonzero_mask] = weight_preliminary_local[nonzero_mask]
                weight_preliminary += weight_preliminary_s
        if not os.path.exists('./Weights/{}/SFP-DP/preliminary/p={}/'.format(name, p)):
            os.makedirs('./Weights/{}/SFP-DP/preliminary/p={}/'.format(name, p))
        cp.savetxt('./Weights/{}/SFP-DP/preliminary/p={}/weight_preliminary.txt'.format(name, p), weight_preliminary, fmt='%.64f', delimiter=' ')

        end = datetime.now()
        run = end - begin
        # print("weight AW Process {}, {} end, info: {}".format(s, cp.count_nonzero(U[s]), end))
        print('SFP-DP-{} Process {} running {} with sum={}, max={} and min={}'.format(name, s, run,
                                                                              cp.sum(cp.sum(weight_preliminary)),
                                                                              cp.max(weight_preliminary),
                                                                              cp.min(weight_preliminary)))

    nonzero_mask_preliminary = weight_preliminary != 0
    weight = cp.zeros_like(weight_preliminary)
    weight[nonzero_mask_preliminary] = sensitivity / cp.sqrt(weight_preliminary[nonzero_mask_preliminary])
    if not os.path.exists('./Weights/{}/SFP-DP/p={}/'.format(name, p)):
        os.makedirs('./Weights/{}/SFP-DP/p={}/'.format(name, p))
    cp.savetxt('./Weights/{}/SFP-DP/p={}/weights-SFP-DP.txt'.format(name, p), weight, fmt='%.64f', delimiter=' ')
    print('SFP-DP with sum={}, max={} and min={}'.format(cp.sum(cp.sum(weight)),
                                                                                  cp.max(weight),
                                                                                  cp.min(weight)))
    return weight

#

# 计算SFP-DP-MMI的权重，与p无关
def caculateWeight_SFPDP_MMI(U, V, W_L, sensitivity, name):
    # 将必要的数组转换为CuPy数组
    U_gpu = cp.array(U)
    V_gpu = cp.array(V)
    W_L_gpu = cp.array(W_L)

    n = W_L_gpu.shape[0]
    m = W_L_gpu.shape[1]



    weight_outer_left = cp.loadtxt('./Weights/{}/weightsPreliminary_left.txt'.format(name), dtype=float, delimiter=' ')
    weight_outer_right = cp.loadtxt('./Weights/{}/weightsPreliminary_right.txt'.format(name), dtype=float,
                                    delimiter=' ')

    weight_MMI_pre = cp.zeros((n, m))
    # weight_MMI_pre = cp.loadtxt('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre.txt'.format(name), dtype=float, delimiter=' ')
    weightsSort = cp.zeros((n, n))

    logging.info(f"{torch.cuda.get_device_properties(0)}")

    for s in range(n):
    # for s in range(265, n):

        weight_MMI_pre_s = cp.zeros((n, m))
        begin = datetime.now()
        print(
            "weights for SFP-DP-MMI running Process {}, {}, {}, begin, info: {}".format(s, weight_outer_left.shape, weight_outer_right.shape,
                                                                   begin))
        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # logging.info(f'Using device: {device}')
        # if device.type == 'cuda':
        #     logging.info(f"{torch.cuda.get_device_properties(0)}")

        for t in range(n):
            if s != t:
                I_st = cp.zeros((n, n))
                I_st[s][t] = -1
                I_st[s][s] = 1

                # 优化后的inner_left计算
                U_s = U_gpu[:, s]
                U_t = U_gpu[:, t]
                V_s = V_gpu[:, s]
                V_t = V_gpu[:, t]
                inner_left = cp.sum(U_s * V_s + U_t * V_t)


                # inner_left = cp.diag(cp.array([U_gpu[j].T @ I_st @ V_gpu[j] for j in range(n)]))

                # # 初始化inner_left数组
                # inner_left = cp.zeros(n)
                # # 计算每个j的对角线元素
                # for j in range(n):
                #     # 提取U_gpu[j]的第s个和第t个元素
                #     U_s = U_gpu[j, s]
                #     U_t = U_gpu[j, t]
                #     # 提取V_gpu[j]的第s个和第t个元素
                #     V_s = V_gpu[j, s]
                #     V_t = V_gpu[j, t]
                #     # 计算对角线元素
                #     # U_gpu[j].T @ I_st @ V_gpu[j] 的对角线元素只依赖于U_gpu[j]的第s行和第t行
                #     # 以及V_gpu[j]的第s列和第t列
                #     inner_left += U_s * V_s + U_t * V_t

                # inner_left_list = cp.array([U_gpu[j].T @ I_st @ V_gpu[j] for j in range(n)])
                # inner_left = cp.diag(inner_left_list)
                # inner_left = cp.einsum('ij,jk,kl->il', U_gpu, I_st, V_gpu).diagonal()
                # inner_left = cp.sum(U_gpu * I_st[:, None] * V_gpu, axis=0).diagonal()

                # weight_inner = U_gpu.T @ I_st @ V_gpu - inner_left
                # 优化的 weight_inner
                weight_inner = cp.matmul(U_gpu.T, cp.matmul(I_st, V_gpu)) - inner_left
                weight_st_pre = weight_outer_left @ weight_inner @ weight_outer_right
                # # 优化的 weight_st_pre
                # weight_st_pre = cp.matmul(weight_outer_left, cp.matmul(weight_inner, weight_outer_right))
                weight_st = weight_st_pre[:n, :n]
                weight_st_w = cp.dot(weight_st, W_L_gpu)

                nonzero_mask = weight_st_w != 0
                # weight_MI_pre_local = cp.zeros_like(weight_st_w)
                # weight_MI_pre_local[nonzero_mask] = weight_st_w[nonzero_mask] ** 2
                weight_MI_pre_local = cp.where(nonzero_mask, weight_st_w ** 2, 0)

                weightsSort[s][t] = cp.mean(weight_MI_pre_local)

                # weight_MMI_pre_s += weight_MI_pre_local
                weight_MMI_pre += weight_MI_pre_local

        # cp.savetxt('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre_{}.txt'.format(name, s), weight_MMI_pre_s,
        #            fmt='%.64f', delimiter=' ')\
        # cp.savetxt('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre.txt'.format(name), weight_MMI_pre,
        #            fmt='%.64f', delimiter=' ')
        cp.save('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre.npy'.format(name), weight_MMI_pre)
        # np.save('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre.txt'.format(name), weight_MMI_pre.get())

       #  weight_MI_pre_s = cp.loadtxt('./weight/weight_AW/MI/primilary/weight_MI_pre_' + str(s) + '.txt', dtype=float, delimiter=' ')
        # weight_MI_pre += weight_MI_pre_s

        end = datetime.now()
        run = end - begin
        print('SFP-DP-MMI-{} Process {} run {} with sum={}, max={} and min={}'.format(name, s, run, cp.sum(cp.sum(weight_MMI_pre)), cp.max(weight_MMI_pre),
              cp.min(weight_MMI_pre)))






    cp.savetxt('./Weights/{}/SFP-DP-MMI/preliminary/weights_pre.txt'.format(name), weight_MMI_pre, fmt='%.64f', delimiter=' ')


    nonzero_mask_pre = weight_MMI_pre != 0
    weight_MMI = cp.zeros_like(weight_MMI_pre)
    weight_MMI[nonzero_mask_pre] = sensitivity / cp.sqrt(weight_MMI_pre[nonzero_mask_pre])

    cp.savetxt('./Weights/{}/SFP-DP-MMI/weights-SFP-DP-MMI.txt'.format(name), weight_MMI, fmt='%.64f', delimiter=' ')
    cp.savetxt('./Weights/{}/weightsSort.txt'.format(name), weightsSort, fmt='%.64f', delimiter=' ')

    # weight_MI = cp.loadtxt('./weight/weight_AW/MI/weight_MI.txt', dtype=float, delimiter=' ')

    return weight_MMI


# 计算SFP-DP-LMLT的权重，与p无关
def caculateWeight_SFPDP_LMLT(U, V, W_L, sensitivity, name):
    # 将必要的数组转换为CuPy数组（假设它们适合放入GPU内存中）
    U_gpu = cp.array(U)
    V_gpu = cp.array(V)
    W_L_gpu = cp.array(W_L)

    n = W_L_gpu.shape[0]
    m = W_L_gpu.shape[1]

    weight_outer_left = cp.loadtxt('./Weights/{}/weightsPreliminary_left.txt'.format(name), dtype=float, delimiter=' ')
    weight_outer_right = cp.loadtxt('./Weights/{}/weightsPreliminary_right.txt'.format(name), dtype=float,
                                    delimiter=' ')
    sortIndex_array = cp.loadtxt('./Weights/{}/sortIndex.txt'.format(name), dtype=int, delimiter=' ')

    weight_preliminary_LMLT = cp.zeros((m, n * (n - 1)))

    for s in range(n):
        begin = datetime.now()
        print(
            "preliminary weights for SFP-DP-LMLT running Process {}, {}, {}, begin, info: {}".format(s, weight_outer_left.shape,
                                                                                        weight_outer_right.shape,
                                                                                        begin))
        for t in range(n):
            if s != t:
                I_st = cp.zeros((n, n))
                I_st[s][t] = -1
                I_st[s][s] = 1

                inner_left_list = cp.array([U_gpu[j].T @ I_st @ V_gpu[j] for j in range(n)])
                inner_left = cp.diag(inner_left_list)
                # inner_left = cp.zeros((n, n))
                # inner_left[j][j] = U[j].T @ I_st @ V[j] for j in range(n)

                weight_inner = U_gpu.T @ I_st @ V_gpu - inner_left
                weight_st_prime = weight_outer_left @ weight_inner @ weight_outer_right
                # print(weight_st_prime[:n, :n].shape)
                weight_st = weight_st_prime[:n, :n]
                # weight_st = weight_st_prime[n:, n:].T
                weight_st_w = cp.dot(weight_st, W_L_gpu)
                q = sortIndex_array[s][t]
                weight_q_LR = cp.sum(weight_st_w ** 2, axis=0)
                weight_preliminary_LMLT[:, q - 1] = weight_q_LR

        end = datetime.now()
        run = end - begin
        print('SFP-DP-LMLT-{} Process {} run {} with sum={}, max={} and min={}'.format(name, s, run,
                                                                                    cp.sum(cp.sum(weight_preliminary_LMLT)),
                                                                                    cp.max(weight_preliminary_LMLT),
                                                                                    cp.min(weight_preliminary_LMLT)))

    cp.savetxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_preliminary_LMLT.txt'.format(name), weight_preliminary_LMLT, fmt='%.64f', delimiter=' ')
    print('weight_preliminary_LMLT with sum={}, max={} and min={}'.format(cp.sum(cp.sum(weight_preliminary_LMLT)), cp.max(weight_preliminary_LMLT),
          cp.min(weight_preliminary_LMLT)))
    #
    weight_preliminary_LMLT_pinv = cp.linalg.pinv(weight_preliminary_LMLT)
    cp.savetxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_preliminary_LMLT_pinv.txt'.format(name), weight_preliminary_LMLT_pinv, fmt='%.64f', delimiter=' ')
    print('weight_preliminary_LMLT_pinv with sum={}, max={} and min={}'.format(cp.sum(cp.sum(weight_preliminary_LMLT_pinv)),
                                                                       cp.max(weight_preliminary_LMLT_pinv),
                                                                       cp.min(weight_preliminary_LMLT_pinv)))

    weight_LMLT_preliminary_j = cp.sum(weight_preliminary_LMLT_pinv, axis=1)
    cp.savetxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_preliminary_j.txt'.format(name), weight_LMLT_preliminary_j, fmt='%.64f', delimiter=' ')
    print('weight_LMLT_preliminary_j with sum={}, max={} and min={}'.format(
        cp.sum(cp.sum(weight_LMLT_preliminary_j)),
        cp.max(weight_LMLT_preliminary_j),
        cp.min(weight_LMLT_preliminary_j)))

    for l in range(len(weight_LMLT_preliminary_j)):
        if weight_LMLT_preliminary_j[l] < 0:
            weight_LMLT_preliminary_j[l] = 0

    weight_LMLT_j_1 = weight_LMLT_preliminary_j ** (1 / 3)
    cp.savetxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_j_1.txt'.format(name), weight_LMLT_j_1, fmt='%.64f', delimiter=' ')
    print('weight_LMLT_j_1 with sum={}, max={} and min={}'.format(cp.sum(weight_LMLT_j_1), cp.max(weight_LMLT_j_1), cp.min(weight_LMLT_j_1)))

    weight_LMLT_j_2 = weight_LMLT_preliminary_j ** (2 / 3)
    cp.savetxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_j_2.txt'.format(name), weight_LMLT_j_2, fmt='%.64f', delimiter=' ')
    print('weight_LMLT_j_2 with sum={}, max={} and min={}'.format(cp.sum(weight_LMLT_j_2), cp.max(weight_LMLT_j_2), cp.min(weight_LMLT_j_2)))


    # 加载数据
    # weight_preliminary_LMLT = cp.loadtxt('./Weights/{}/SFP-DP-LMLT/preliminary/weight_preliminary_LMLT.txt'.format(name), dtype=float)
    # weight_preliminary_LMLT_pinv = cp.loadtxt(
    #     './Weights/{}/SFP-DP-LMLT/preliminary/weight_preliminary_LMLT_pinv.txt'.format(name), dtype=float)
    # weight_LMLT_preliminary_j = cp.loadtxt(
    #     './Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_preliminary_j.txt'.format(name), dtype=float)
    # for l in range(len(weight_LMLT_preliminary_j)):
    #     if weight_LMLT_preliminary_j[l] < 0:
    #         weight_LMLT_preliminary_j[l] = 0
    # weight_LMLT_j_1 = cp.loadtxt(
    #     './Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_j_1.txt'.format(name), dtype=float)
    # weight_LMLT_j_2 = cp.loadtxt(
    #     './Weights/{}/SFP-DP-LMLT/preliminary/weight_LMLT_j_2.txt'.format(name), dtype=float)

    weight_LMLT = cp.zeros((n, n))
    for s in range(n):
        begin = datetime.now()
        print("weights for SFP-DP-LMLT running Process {}, begin, info: {}".format(s, begin))
        for t in range(n):
            if s != t:
                q = sortIndex_array[s][t]
                weight_left = cp.power(sensitivity, 2 / 3) * cp.sum(weight_LMLT_j_1 )
                weight_right = 0
                for j in range(m):
                    if weight_LMLT_j_2[j] != 0 and weight_preliminary_LMLT_pinv[q - 1][j] * cp.power(sensitivity, 2 / 3) / \
                            weight_LMLT_j_2[j] > 0:
                        weight_right += (weight_preliminary_LMLT_pinv[q - 1][j] * cp.power(sensitivity, 2 / 3) /
                                         weight_LMLT_j_2[j]) ** (1 / 2)
                # print(weight_left, weight_right)
                weight_LMLT[s, t] = cp.power(n, 3 / 2) * weight_left * weight_right

        end = datetime.now()
        run = end - begin
        print('SFP-DP-LMLT-{} Process {} run {} with sum={}, max={} and min={}'.format(name, s, run,
                                                                                    cp.sum(cp.sum(weight_LMLT)),
                                                                                    cp.max(weight_LMLT),
                                                                                    cp.min(weight_LMLT)))

    cp.savetxt('./Weights/{}/SFP-DP-LMLT/weight_LMLT.txt'.format(name), weight_LMLT, fmt='%.64f', delimiter=' ')
    print('weight_LMLT with sum={}, max={} and min={}'.format(cp.sum(weight_LMLT), cp.max(weight_LMLT), cp.min(weight_LMLT)))

    return weight_LMLT
















