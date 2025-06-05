import gc
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from .datasets import MNIST_truncated, MNIST_rotated, CIFAR10_truncated, CIFAR10_rotated, CIFAR100_truncated, \
    SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, \
    ImageFolder_custom, USPS_truncated, logger, WM811K_custom
from math import sqrt
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import time
import random
import copy
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file
from PIL import Image
import pandas as pd
import cv2
import pickle


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_usps_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    usps_train_ds = USPS_truncated(datadir, train=True, download=True, transform=transform)
    usps_test_ds = USPS_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = usps_train_ds.data, usps_train_ds.target
    X_test, y_test = usps_test_ds.data, usps_test_ds.target

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, y_train, X_test, y_test)


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_mnist_rotated_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


# def preprocess_wm811k():
#     df = pd.read_pickle("OTDD/generate_data/data/wm811k/data/MIR-WM811K/Python/WM811K.pkl")
#     # get the failure wafer map
#     failure_type = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
#     df_failure = df[df['failureType'].isin(failure_type)].loc[:, ['waferMap', 'failureType', 'trainTestLabel']]
#
#     # transform the failure_type into digital label
#     group_labels = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
#                     'Near-full': 7}
#     df_failure['Label'] = df_failure['failureType'].map(group_labels)
#
#     # Select training and test data
#     trainData_temp = df_failure[df_failure['trainTestLabel'] == 'Training'].reset_index()  # 训练集：17625
#     testData_temp = df_failure[df_failure['trainTestLabel'] == 'Test'].reset_index()  # 测试集：7894
#
#     # obtain the dataset for federated training
#     trainData = trainData_temp.loc[:, ['waferMap', 'Label']]
#     testData = testData_temp.loc[:, ['waferMap', 'Label']]
#
#     del df
#     del failure_type
#     del df_failure
#     del group_labels
#     del trainData_temp
#     del testData_temp
#     gc.collect()
#
#     return trainData, testData
#
#
# def image_resize(images):
#     # preprocess the shape of data to the size of (36, 36)
#     TARGET_SIZE = (36, 36)
#     images_ = []
#     for img in images:
#         image = cv2.resize(img / img.max(), dsize=(TARGET_SIZE[0], TARGET_SIZE[1]), interpolation=cv2.INTER_CUBIC)
#         images_.append(image)
#     return np.asarray(images_).astype("float32")

def load_wm811k_data():
    # trainData, testData = preprocess_wm811k()
    with open("data/wm811k/data/MIR-WM811K/Python/wm811k-processed.pkl", "rb") as file:
        wm811k = pickle.load(file)
    X_train = wm811k["X_train"]
    y_train = wm811k["y_train"]
    X_test = wm811k["X_test"]
    y_test = wm811k["y_test"]

    # X_train = image_resize(trainData.iloc[:, 0].values)
    # y_train = trainData.iloc[:, 1].values
    # X_test = image_resize(testData.iloc[:, 0].values)
    # y_test = testData.iloc[:, 1].values

    # 清除内存
    # del trainData
    # del testData
    # gc.collect()
    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=False, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_rotated_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_rotated(datadir, rotation=0, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_rotated(datadir, rotation=0, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    print(datadir)
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    xray_train_ds = ImageFolder_custom(datadir, train=True, transform=transform_train)
    xray_test_ds = ImageFolder_custom(datadir, train=False, transform=transform_test)

    # X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    # X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])
    X_train, y_train = xray_train_ds.data, xray_train_ds.target
    X_test, y_test = xray_test_ds.data, xray_test_ds.target

    # y_test_sorted = np.sort(y_test)
    return (X_train, y_train, X_test, y_test)


def load_celeba_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train = celeba_train_ds.attr[:, gender_index:gender_index + 1].reshape(-1)
    y_test = celeba_test_ds.attr[:, gender_index:gender_index + 1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)


def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    # logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, local_view=False):
    np.random.seed(1234)
    torch.manual_seed(1234)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'wm811k':
        X_train, y_train, X_test, y_test = load_wm811k_data()
    elif data == 'mnist_rotated':
        X_train, y_train, X_test, y_test = load_mnist_rotated_data(datadir)
    elif dataset == 'usps':
        X_train, y_train, X_test, y_test = load_usps_data(datadir)
    elif dataset == 'cifar10_rotated':
        X_train, y_train, X_test, y_test = load_cifar10_rotated_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0, 3999, 4000, dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    # elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('rcv1', 'SUSY', 'covtype'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == 'covtype':
            y_train = y_train - 1
        else:
            y_train = (y_train + 1) / 2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    elif dataset in ('a9a'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    n_train = y_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "Dir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset in ('cifar100'):
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'wm811k':
            K = 8

        N = y_train.shape[0]
        # np.random.seed(2021)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))

                if dataset == 'wm811k' and beta == 0.1:
                    sorted_indices = np.argsort([len(idx_j) for idx_j in idx_batch])  # 从小到大排序
                    sorted_proportions = np.sort(proportions)[::-1]  # 从大到小排序，排列顺序不影响狄利克雷分布
                    proportions = np.zeros_like(proportions)
                    for i, idx in enumerate(sorted_indices):
                        proportions[idx] = sorted_proportions[i]

                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                # np.cumsum(proportions)的作用是计算每个客户端应该选择的样本数量的累加和，设置划分断点，以便后续计算每个客户端应该选择的样本数量。
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > 'noniid1' and partition <= 'noniid9':
        num = eval(partition[6:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100':
            num = num * 10
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            num = K * (num * 10 / 100)
        elif dataset == 'wm811k':
            K = 8
        else:
            K = 10

        print(f'K: {K}')
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]
            contain = []
            for i in range(n_parties):
                current = [i % K]  # 通过一个循环，生成 n_parties 个子集，其中每个子集都以 i % K（i 对 K 取余）的类别作为起始
                times[i % K] += 1  # 记录每个类别已经被分配的次数
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)  # 在每个子集中逐个添加其他类别，直到达到所需的类别数量num
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]  # 返回类别为i的样本索引
                np.random.shuffle(idx_k)
                # print(f"class {i}: {idx_k}, times: {times[i]}")
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1
                        # 新加代码
            #             if len(net_dataidx_map[j]) % 10 != 0:
            #                 idx_needed = 10 - (len(net_dataidx_map[j]) % 10)
            #                 idx_ext = np.random.choice(idx_k, size=idx_needed, replace=False)
            #                 net_dataidx_map[j] = np.append(net_dataidx_map[j], idx_ext)
            # # 新加代码
            # for i in range(K):
            #     idx_k = np.where(y_train == i)[0]  # 返回类别为i的样本索引
            #     np.random.shuffle(idx_k)
            #     for j in range(n_parties):
            #         if len(net_dataidx_map[j]) % 100 != 0:
    elif partition > 'noniid11' and partition <= 'noniid19':
        print('Modified Non-IID partitioning')
        num = eval(partition[7:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'wm811k':
            K = 8
        else:
            K = 10

        print(f'Dataset {dataset}, K: {K}, {partition}')
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]  # 记录每个标签在数据划分中出现的次数
            contain = []

            # aa = np.random.randint(low=0, high=K, size=num)
            aa = np.random.choice(np.arange(K), size=num, replace=False)  # 从K个标签中随机选择num个标签，并分配给参与方
            remain = np.delete(np.arange(K), aa)
            # print(f'Client 0 , {len(aa)}')
            # print(f'Unique a {len(np.unique(aa))}')
            # print(f'Unique remain {len(np.unique(remain))}')
            contain.append(copy.deepcopy(aa.tolist()))
            for el in aa:
                times[el] += 1

            for i in range(n_parties - 1):
                x = np.random.randint(low=int(np.ceil(K / 2)), high=K)
                y = np.random.randint(low=0, high=int(K / 4) + 1)

                rand = np.random.choice([0, 1, 2], size=1, replace=False)
                # print(rand)
                if rand == 0 or rand == 1:  # 将num个标签均匀的分配到x个参与方中
                    s = int(np.ceil((x / K) * num))
                    if s == num and rand == 0:
                        s = s - int(np.ceil(0.05 * num))  # 防止某些参与方的标签数量过多
                elif rand == 2:
                    s = int(np.ceil((y / K) * num))  # 将num个标签分配到y个参与方中

                labels = np.random.choice(aa, size=s, replace=False).tolist()
                # print(f'Client {i} , {len(labels)}, S {s}')
                # print(labels)
                labels.extend(np.random.choice(remain, size=(num - s), replace=False).tolist())
                # print(f'Client {i+1} , {len(labels)}')
                # ccc = np.unique(labels)
                # print(f'Client {i+1} , {len(ccc)}')

                for el in labels:
                    times[el] += 1
                contain.append(labels)
                # print(len(labels))

            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                # print(f'{i}: {times[i]}')
                split = np.array_split(idx_k, times[i])  # 标签i出现多少次，就将标签为i的样本划分为多少个array
                # print(f'len(split) {len(split)}, times[i] {times[i]}')
                ids = 0
                for j in range(n_parties):
                    # print(f'Client {i}, {len(contain[j])}')
                    if i in contain[j]:  # 如果标签i在客户端j中，就将标签i划分的array组中的一个样本集分配给第j个客户端
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user + 1, dtype=np.int32)
        for i in range(1, num_user + 1):
            user[i] = user[i - 1] + u_train[i - 1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i: np.zeros(0, dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i] = np.append(net_dataidx_map[i], np.arange(user[j], user[j + 1]))

    print(f'partition: {partition}')
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    print('Data statistics Train:\n %s \n' % str(traindata_cls_counts))

    if local_view:
        net_dataidx_map_test = {i: [] for i in range(n_parties)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test == l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else:
        net_dataidx_map_test = None
        testdata_cls_counts = None

    return (
    X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    model.to(device)

    w = model.state_dict()
    name = list(w.keys())[0]
    print(f'COMP ACC {w[name][0, 0, 0]}')

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                # pred = out.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:, row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0,
                   dataidxs_test=None,
                   same_size=False, target_transform=None, rotation=0):
    if dataset in ('mnist', 'mnist_rotated', 'femnist', 'fmnist', 'cifar10', 'cifar10_rotated', 'cifar100',
                   'svhn', 'tinyimagenet', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY', 'usps', 'wm811k'):
        if dataset == 'mnist' or dataset == 'mnist_rotated':
            if dataset == 'mnist':
                dl_obj = MNIST_truncated
            elif dataset == 'mnist_rotated':
                dl_obj = MNIST_rotated

            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated

            if same_size:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, total),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        elif dataset == 'cifar10' or dataset == 'cifar10_rotated':
            if dataset == 'cifar10':
                dl_obj = CIFAR10_truncated
            elif dataset == 'cifar10_rotated':
                dl_obj = CIFAR10_rotated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: F.pad(
                #    Variable(x.unsqueeze(0), requires_grad=False),
                #    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])

        elif dataset == 'tinyimagenet':
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: np.array(x)),  # 将 PIL 图像转换为 NumPy 数组
                transforms.Lambda(lambda x: x.astype(np.float32) / 255.0),  # 归一化到 [0, 1]
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])
            transform_test = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: np.array(x)),  # 将 PIL 图像转换为 NumPy 数组
                transforms.Lambda(lambda x: x.astype(np.float32) / 255.0),  # 归一化到 [0, 1]
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])

            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train)
            test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=False, shuffle=False)

        elif dataset == 'wm811k':
            dl_obj = WM811K_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.46280908584594727,), (0.2907727062702179,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.45607784390449524,), (0.2872909605503082,))
            ])

            train_ds = dl_obj(dataidxs=dataidxs, train=True, transform=transform_train)
            test_ds = dl_obj(dataidxs=dataidxs_test, train=False, transform=transform_test)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


        elif dataset == 'usps':
            dl_obj = USPS_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None

        if dataset != 'tinyimagenet' and dataset != 'wm811k':
            if dataset == 'mnist_rotated' or dataset == 'cifar10_rotated':
                train_ds = dl_obj(datadir, rotation=rotation, dataidxs=dataidxs, train=True, transform=transform_train,
                                  target_transform=target_transform, download=True)
                test_ds = dl_obj(datadir, rotation=rotation, dataidxs=dataidxs_test, train=False,
                                 transform=transform_test,
                                 target_transform=target_transform, download=True)
            else:
                train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                                  target_transform=target_transform, download=True)
                test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test,
                                 target_transform=target_transform, download=True)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if (n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if (n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if (n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if (n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


def group_by_value(my_dict):
    grouped = defaultdict(list)
    for key, value in my_dict.items():
        grouped[value].append(key)
    result = list(grouped.values())
    return result


def norm(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
        x = x.astype('float32') / 255
        m = x.mean(axis=(0, 1, 2))
        s = x.std(axis=(0, 1, 2))
        x = (x - m) / s
        x = torch.from_numpy(x)
    else:
        x = x.astype('float32') / 255
        m = x.mean(axis=(0, 1, 2))
        s = x.std(axis=(0, 1, 2))
        x = (x - m) / s
    return x
