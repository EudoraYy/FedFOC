import copy

import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import hdbscan
from scipy.spatial.distance import jaccard
from collections import Counter

def fcm_clustering(sim, n_clusters=3, m=2):
    sim_arr = np.array(sim)
    # cntr: 聚类中心矩阵
    # u: 隶属度矩阵
    # u0: 初始隶属度矩阵
    # d: 数据点到聚类中心的距离矩阵
    # jm: 目标函数值的数组，目标函数表示隶属度的改善程度。
    # p: 模糊指数（迭代优化后的）
    # fpc: 模糊分割系数（Fuzzy Partition Coefficient），用于评估聚类结果的模糊性，取值范围为[1.1,10]
    # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(sim_arr.T, n_clusters, m, error=0.005, maxiter=1000, distance=custom_distance)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(sim_arr.T, n_clusters, m, error=0.005, maxiter=1000)
    # print("u.shape:{}".format(u.shape))
    cluster_membership = np.argmax(u, axis=0)
    clusters = cluster_membership.tolist()

    # 将小于0.3的隶属度归0，并转置矩阵
    w_aft = np.zeros([n_clusters, len(clusters)])
    for i, l in enumerate(u):
        for j, p in enumerate(l):
            if p > 0.3:
                w_aft[i][j] = p
            else:
                w_aft[i][j] = 0
    # 获w_aft中的每个客户端的隶属度之和
    s = np.zeros(len(clusters))
    for i, l in enumerate(w_aft):
        for j, p in enumerate(l):
            s[j] += w_aft[i][j]
    # 更新隶属度矩阵
    w_fine = np.zeros([n_clusters, len(clusters)])
    for i, l in enumerate(w_aft):
        for j, p in enumerate(l):
            w_fine[i][j] = p / s[j]
    # 获取聚类结果
    clusters_aft = []
    for k in range(n_clusters):
        c_list = [i for i, element in enumerate(w_fine[k]) if element != 0]
        clusters_aft.append(c_list)

    # plt.figure(figsize=(12, 8))
    # y = np.ones(len(sim_T)).reshape(-1, 1)
    # plt.scatter(sim_T, y, c=cluster_membership)
    # plt.axis('off')
    # # plt.savefig("/home/yyzhao/PACFL-main/OTDD/scripts/pic/cluster_overlap_"+str(n_clusters)+".png", bbox_inches='tight')
    # plt.show()
    return clusters_aft, w_fine


def gmm_clustering(sim, n_components=2):
    sim_T = np.array(sim).reshape(-1, 1)
    gmm = GaussianMixture(n_components)
    gmm.fit(sim_T)
    labels = gmm.predict(sim_T)
    membership_scores = gmm.predict_proba(sim_T)

    # 将小于0.1的隶属度归0，并转置矩阵
    w_aft = np.zeros([len(set(labels)), len(labels)])
    for i, l in enumerate(membership_scores):
        for j, p in enumerate(l):
            if p > 0.3:
                w_aft[j][i] = p
            else:
                w_aft[j][i] = 0
    # 获w_aft中的每个客户端的隶属度之和
    s = np.zeros(len(labels))
    for i, l in enumerate(w_aft):
        for j, p in enumerate(l):
            s[j] += w_aft[i][j]
    # 更新隶属度矩阵
    w_fine = np.zeros([len(set(labels)), len(labels)])
    for i, l in enumerate(w_aft):
        for j, p in enumerate(l):
            w_fine[i][j] = p / s[j]
    # 获取聚类结果
    clusters = []
    for k in range(len(set(labels))):
        c_list = [i for i, element in enumerate(w_fine[k]) if element != 0]
        clusters.append(c_list)
    return clusters, w_fine


def hdbscan_clustering(sim, min_size=10):
    sim_T = np.array(sim).reshape(-1, 1)
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters_hdb = hdbscan_clusterer.fit_predict(sim_T)
    # 获取隶属度
    degree_mem = np.zeros([(len(set(clusters_hdb))-1), len(clusters_hdb)])
    for ii, c in enumerate(clusters_hdb):
        if c >= 0:
            degree_mem[c][ii] = 1
        else:
            fcm_clusters, fcm_u = fcm_clustering(sim, n_clusters=3, m=2)
            degree_mem[:, ii] = fcm_u[:, ii]
    print(degree_mem)
    # 获取聚类结果
    clusters = []
    for k in range((len(set(clusters_hdb))-1)):
        c_list = [i for i, element in enumerate(degree_mem[k]) if element != 0]
        clusters.append(c_list)
    return clusters, degree_mem


def initialize_membership(n_points, n_clusters):
    # 初始化隶属度矩阵
    random_matrix = np.random.rand(n_points, n_clusters)
    sums = np.sum(random_matrix ** 2, axis=1, keepdims=True)
    normalized_matrix = (random_matrix ** 2) / sums
    return normalized_matrix


def fcm_revise(data, n_clusters, custom_distance, num_vote=2, m=2, max_iter=1000, epsilon=1e-5): # 20240819
    n_points = len(data)
    num_vote = num_vote
    membership = initialize_membership(n_points, n_clusters)
    for _ in range(max_iter):
        # 更新距离值向量的簇中心
        cluster_centers = np.zeros((n_clusters, num_vote+1))
        for z in range(n_clusters):
            threshold = np.percentile(membership[:, z], 0)
            cluster_data = [data[k] for k in range(n_points) if membership[k][z] > threshold]
            if len(cluster_data) > 0:
                cluster_centers[z][num_vote] = np.sum([membership[k][z] ** m * data[k][len(data[0])-1]
                               for k in range(n_points) if membership[k][z] > threshold]) / \
                                np.sum([membership[k][z] ** m for k in range(n_points)
                                        if membership[k][z] > threshold])
                # for k in range(n_points):
                #     cluster_centers[z][num_vote] += membership[k, z] ** m * data[k][len(data[0])]
                # cluster_centers[z][num_vote] /= np.sum(membership ** m, axis=0)[z]

                # 特殊情况处理：对于前两维，取最频繁的值
                # most_common = []
                # if num_vote == 2:
                #     for idx in range(len(data[0])):
                #         most_common_temp = Counter(data[i][idx] for i in range(n_points) if membership[i, z] > 0).most_common(1)[0][
                #             0]
                #         most_common.append(most_common_temp)
                most_common = None
                if num_vote >= 2:
                    cluster_values = [data[client_id] for client_id in range(n_points)
                                      if membership[client_id, z] > threshold]
                    all_values = [value for client_data in cluster_values for value in client_data[:len(data[0])-1]]
                    # for client_id in range(n_points):
                    #     client_values = [data[client_id][vv] for vv in range(len(data[0]))]
                    #     all_values.extend(client_values)
                    most_common_temp = Counter(all_values).most_common(num_vote)
                    compl = []
                    if len(most_common_temp) < num_vote:
                        for _ in range(num_vote - len(most_common_temp)):
                            compl.append(0)
                        most_common = [item[0] for item in most_common_temp]+compl
                    else:
                        most_common = [item[0] for item in most_common_temp]
                    # print("MinHash的簇中心为: {}".format(most_common))
                else:
                    raise ValueError('投票阈值设置错误！')
                cluster_centers[z, :num_vote] = most_common
            else:
                continue
        #print("cluster_centers:{}".format(cluster_centers))

        # 更新隶属度
        new_membership = np.zeros_like(membership)
        for i in range(n_points):
            for j in range(n_clusters):
                sum_memb = 0
                for k in range(n_clusters):
                    if custom_distance(data[i], cluster_centers[k]) == 0:
                        new_membership[i, j] = 1.0
                        continue
                    else:
                        sum_memb += (custom_distance(data[i], cluster_centers[j]) /
                                     custom_distance(data[i], cluster_centers[k])) ** (2 / (m - 1))
                new_membership[i, j] = 1 / sum_memb

                # 检查收敛
        if np.allclose(membership, new_membership, atol=epsilon):
            break

        membership = new_membership
        #print(membership)

    cluster_membership = np.argmax(membership, axis=1)
    # print(cluster_membership)
    clusters = cluster_membership.tolist()
    # print(clusters)

    # 将小于0.3的隶属度归0，并转置矩阵
    w_aft = np.zeros([n_points, n_clusters])
    for i, l in enumerate(membership):
        for j, p in enumerate(l):
            if p > 1 / n_clusters:
                w_aft[i][j] = p
            else:
                w_aft[i][j] = 0
    # 获w_aft中的每个客户端的隶属度之和
    # print(w_aft)
    s = np.sum(w_aft, axis=1)
    # print(s)

    # 更新隶属度矩阵
    new_membership_2 = np.zeros_like(membership)
    for i, l in enumerate(w_aft):
        for j, p in enumerate(l):
            new_membership_2[i][j] = p / s[i]
    # print(new_membership_2)

    # 获取聚类结果
    cluster_of_client = []
    for k in range(n_clusters):
        c_list = [i for i, element in enumerate(new_membership_2[:, k]) if element != 0]
        cluster_of_client.append(c_list)

    u = new_membership_2.T


    return cluster_of_client, u




