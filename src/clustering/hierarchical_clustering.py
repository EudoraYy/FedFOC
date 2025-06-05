import numpy as np
from typing import Iterable
import copy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import os
import random
from collections import Counter


def flatten(items):  # 生成器函数，可用next()逐个生成序列中的元素
    # 功能是将任意嵌套的可迭代对象展开为一个一维的列表元素（不是列表）
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def calculating_adjacency(clients_idxs, U):
    nclients = len(clients_idxs)

    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            # print(idx1)
            # print(U)
            # print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])

            # sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            # sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            # sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            # 下行代码的作用是将矩阵U1的转置和矩阵U2的矩阵乘积矩阵mul中的元素限制在[-1.0,1.0]之间
            mul = np.clip(U1.T @ U2, a_min=-1.0, a_max=1.0)
            # 下行代码计算两个向量之间的夹角，并将其转换为角度制，然后将其作为相似度存储在一个相似度矩阵对应的位置（相似度越大，夹角越小）
            sim_mat[idx1, idx2] = np.min(np.arccos(mul)) * 180 / np.pi

    return sim_mat


def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix.

    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)

    :return: clusters
    '''

    label_assg = {i: i for i in range(len(A))}

    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A, -np.NINF)
        # print(f'step {step} \n {A}')
        step += 1
        # np.unravel_index是将A中的最小值在一维数组中的索引值按照A的形状转换为最小值所在的行列坐标
        ind = np.unravel_index(np.argmin(A, axis=None), A.shape)

        if len(label_assg) == 3:  # 如果A中的最小值都大于阈值，则停止聚类  A[ind[0], ind[1]] > thresh / len(label_assg) == 3
            print('Breaking HC')
            break
        else:  # 否则将最小值所在的行和列合并，并更新A和label_assg
            np.fill_diagonal(A, 0)
            if linkage == 'maximum':
                Z = np.maximum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == 'minimum':
                Z = np.minimum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == 'average':
                Z = (A[:, ind[0]] + A[:, ind[1]]) / 2

            A[:, ind[0]] = Z
            A[:, ind[1]] = Z
            A[ind[0], :] = Z
            A[ind[1], :] = Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) == list:
                label_assg[ind[0]].append(label_assg[ind[1]])
            else:
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]  # label_assg用于存储每个数据点所属的簇类id

            label_assg.pop(ind[1], None)  # 删除键为ind[1]的键值对，若该键不存在则返回None

            temp = []
            for k, v in label_assg.items():  # 更新删除某列后的A的键
                if k > ind[1]:
                    kk = k - 1
                    vv = v
                else:
                    kk = k
                    vv = v
                temp.append((kk, vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))  # list(label_assg[k])用于将label_assg[k]转换为一维元素列表
        elif type(label_assg[k]) == int:
            clusters.append([label_assg[k]])

    return clusters


def hierarchical_clustering_mix(data, distance_fn, num_clusters):
    clusters = [{i} for i, _ in enumerate(data)]
    centroids = [copy.deepcopy(data[i]) for i in range(len(data))]  # 初始质心就是数据点本身
    distances = [[0] * len(data) for _ in data]

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[i][j] = distances[j][i] = distance_fn(data[i], data[j])
    # print(distances)

    while len(clusters) > num_clusters:
        # 每次只找一个最小值及其坐标，直到簇的数量满足要求
        closest_pair = min(((distances[i][j], i, j) for i in range(len(clusters))
                            for j in range(i + 1, len(clusters)) if (distances[i][j] >= 0 and i != j)),
                           key=lambda x: x[0])  # key=lambda x: x[0]是个匿名函数，表示比较数据中的第一个元素，例如：(0.1488494873046875, 0, 1)
        merge_idx1, merge_idx2 = closest_pair[1], closest_pair[2]  # 找到最相似的两组数据的索引
        clusters[merge_idx1].update(
            clusters[merge_idx2])  # 将merge_idx2指向的聚类合并到merge_idx1指向的聚类中，例如：[{0, 1}, {1}, {2}, {3}, {4}]
        print('合并第{}和第{}个簇，最小距离是：{}；合并之后为：{}'.format(merge_idx1, merge_idx2,
                                                                      distances[merge_idx1][merge_idx2], clusters))

        # 更新簇中心
        dim = len(data[0]) - 1
        merge_data = []
        temp_O = 0
        for idx in clusters[merge_idx1]:
            temp_O += data[idx][dim]
            centroids[merge_idx1][dim] = temp_O / len(clusters[merge_idx1])
            merge_data.append(data[idx])
        # 特殊情况处理：对于前两维，取最频繁的值
        print("合并的数据为：{}".format(merge_data))
        center = []
        for idx in range(dim):
            temp = []
            for row in merge_data:
                temp.append(row[idx])
            most_common = Counter(temp).most_common(1)[0][0]
            center.append(most_common)
        centroids[merge_idx1][:dim] = center

        # 删除已被合并的簇
        del clusters[merge_idx2]
        centroids.pop(merge_idx2)
        distances.pop(merge_idx2)
        for row in distances:
            row.pop(merge_idx2)

        # 更新距离矩阵，将merge_idx2指向的聚类合并到merge_idx1指向的聚类中。
        for k in range(len(clusters)):
            if k != merge_idx1:
                distances[merge_idx1][k] = distance_fn(centroids[merge_idx1], centroids[k])
                distances[k][merge_idx1] = distances[merge_idx1][k]

    clusters_list = [list(item) for item in clusters]
    return clusters_list


def hierarchical_clustering_otdd(A, thresh=3.0, lk='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix.

    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)

    :return: clusters
    '''
    # 计算距离矩阵
    # A_norm = (A - np.min(A)) / ((np.max(A) - np.min(A)))
    # A_arr = np.array(A)
    dist_matrix = pdist(A, metric='euclidean')
    # 进行层次聚类
    Z = linkage(dist_matrix, method=lk)
    thresh_set = round(random.uniform(Z[:, 2][-3], Z[:, 2][-2]), 4)
    print("层次聚类最大的三个类间距离为：{}, {}, {}".format(Z[:, 2][-1], Z[:, 2][-2], Z[:, 2][-3]))
    print("层次聚类的阈值为：{}".format(thresh_set))
    # 绘制树状图
    # plt.figure(figsize=(16, 10), dpi=80)
    # plt.title('Clients Dendrogram', fontsize=22)
    # dendrogram(Z, labels=range(len(A)), color_threshold=thresh_set)
    # plt.xticks(fontsize=12)
    # dir = os.path.abspath("/home/yyzhao/PACFL-main/OTDD/save_results/otdd/pic/all_data/dendrogram"
    # +str(thresh_set)+".png")
    # plt.savefig(dir, dpi=80, bbox_inches='tight')
    # plt.show()
    clusters = fcluster(Z, t=thresh_set, criterion='distance')

    return clusters


def error_gen(actual, rounded):
    divisor = np.sqrt(1.0 if actual < 1.0 else actual)
    return abs(rounded - actual) ** 2 / divisor


def round_to(percents, budget=100):  # 将分配给客户端的预算进行四舍五入，以确保预算总和等于budget
    if not np.isclose(sum(percents), budget):
        raise ValueError
    n = len(percents)
    rounded = [int(x) for x in percents]
    up_count = budget - sum(rounded)
    # 计算每个类别分配预算时的误差，并将这些误差按升序排序
    errors = [(error_gen(percents[i], rounded[i] + 1) - error_gen(percents[i], rounded[i]), i) for i in range(n)]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1  # 返回的是一个列表，其中每个元素表示每个类别分配的预算的整数值
    return rounded
