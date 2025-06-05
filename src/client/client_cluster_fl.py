import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F


class Client_ClusterFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None, train_ds_local=None, test_ds_local=None,
                 num_clusters=3, num_samples=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.lds_train = train_ds_local
        self.lds_test = test_ds_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.num_clusters = num_clusters
        self.num_samples = num_samples

        self.importance_estimated = []

    def estimate_importance_weights(self, cluster_vec, num_classes, count_smoother=0.0001):
        import time
        start_time = time.time()
        # fedsoft
        with torch.no_grad():  # 所得到的tensor的requires_grad都自动设置为False
            table = np.zeros((self.num_clusters, self.num_samples))
            start_idx = 0
            nst_cluster_sample_count = [0] * self.num_clusters
            for x, y in self.ldr_train:
                x = x.to(self.device)
                y = y.to(self.device)
                for s, cluster in enumerate(cluster_vec):
                    cluster.eval()
                    out = cluster(x).view(-1, num_classes)
                    loss = self.loss_func(out, y)
                    table[s][start_idx:start_idx + len(x)] = loss.cpu()
                start_idx += len(x)
            min_loss_idx = np.argmin(table, axis=0)
            for s in range(self.num_clusters):
                nst_cluster_sample_count[s] += np.sum(min_loss_idx == s)
            for s in range(self.num_clusters):
                if nst_cluster_sample_count[s] == 0:
                    nst_cluster_sample_count[s] = count_smoother * self.num_samples
            self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])
        end_time_cluster = time.time()
        elapsed_time_cluster = end_time_cluster - start_time
        print("客户端的集群身份判断时间为：{}".format(elapsed_time_cluster))

    def train(self, alg, cluster_vec=None, reg_weight=0.01):
        self.net.to(self.device)
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                if alg == 'fedsoft':
                    mse_loss = nn.MSELoss(reduction='sum')
                    for i, cluster in enumerate(cluster_vec):
                        l2 = None
                        for (param_local, param_cluster) in zip(self.net.parameters(), cluster.parameters()):
                            if l2 is None:
                                l2 = mse_loss(param_local, param_cluster)
                            else:
                                l2 += mse_loss(param_local, param_cluster)
                        loss += reg_weight / 2 * self.importance_estimated[i] * l2
                else:
                    pass
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.net.eval()

        #         if self.save_best:
        #             _, acc = self.eval_test()
        #             if acc > self.acc_best:
        #                 self.acc_best = acc

        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def get_importance(self, count=True):  # fedsoft
        if count:
            return [ust * self.num_samples for ust in self.importance_estimated]
        else:
            return self.importance_estimated

    def get_client_labels(self):
        return np.array(self.lds_train.target)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        # print(f"correct: {correct}")
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy