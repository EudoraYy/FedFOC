import copy
import torch
from torch import nn

def FedAvg(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    global w_avg
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        # w_avg[k] = torch.div(w_avg[k].cuda(), len(w))

    return w_avg


def FedAvg_model(models, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    global model_avg
    if weight_avg == None:
        weight_avg = [1/len(models) for i in range(len(models))]

    model_avg = copy.deepcopy(models[0])
    for w, c_model in zip(weight_avg, models):
        for server_param, client_param in zip(model_avg.parameters(), c_model.parameters()):
            server_param.data += client_param.data.clone() * w
    return model_avg