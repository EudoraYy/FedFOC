from src.data import *
from src.models import *
from src.fedavg import *
from src.client import *
from src.clustering import *
from src.utils import *
from OTDD.otdd_pytorch import TensorDataset, PytorchEuclideanDistance
from OTDD.otdd_pytorch import SinkhornTensorized, SamplesLossTensorized
from src.clustering import overlapping_clustering
from  datasketch import MinHash
import pickle
import psutil
import math

print('The available package import was successful!')

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.init()
torch.cuda.set_device(args.gpu)  ## Setting cuda on GPU


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_data(path_):
    with open(path_, 'rb') as f:
        idxs_data = pickle.load(f)
    return idxs_data


def norm(A):
    if isinstance(A, torch.Tensor):
        A_norm = (A - torch.min(A)) / (torch.max(A) - torch.min(A))
    else:
        A_norm = (A - np.min(A)) / (np.max(A) - np.min(A))
    return A_norm

# 自定义聚类距离函数
def custom_distance(x1, x2, jaccard_weights=args.weight_distance, euclidean_weights=(1-args.weight_distance)):
    # Both x and y are tuples that contain the distance and the feature vector
    dim1 = len(x1)-1
    dim2 = len(x2)-1
    jaccard_dist = 1 - (len(set(x1[:dim1]) & set(x2[:dim2])) / len(set(x1[:dim1]) | set(x2[:dim2])))  # x[:2] and y[:2] are MinHash signatures
    euclidean_dist = math.sqrt((x1[dim1] - x2[dim2])**2)  # x1[dim1] and x2[dim2] are distance values
    return math.sqrt(jaccard_weights * jaccard_dist ** 2 + euclidean_weights * euclidean_dist ** 2)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, LR {}, Ep {}, Rounds {}, frac {}, num_perm {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition+str(args.beta), args.lr,
                    args.local_ep, args.rounds, args.frac, args.num_perm)
print(s)
print(str(args))

# #################################### Client data loading
net_dataidx_map, net_dataidx_map_test = {i: None for i in range(args.num_users)}, {j: None for j in range(args.num_users)}
traindata_cls_counts, testdata_cls_counts = {i: None for i in range(args.num_users)}, {j: None for j in range(args.num_users)}

if args.partition == 'Dir':
    _path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition+str(args.beta))
    path = args.savedir + args.alg + '/' + args.partition+str(args.beta) + '/' + args.dataset + '/'
    mkdirs(path)
else:
    _path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition)
    path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
    mkdirs(path)

for i in range(args.num_users):
    net_dataidx_map[i] = load_data(os.path.join(_path, "train/", "task_{}".format(i), "train.pkl"))
    net_dataidx_map_test[i] = load_data(os.path.join(_path, "test/", "task_{}".format(i), "test.pkl"))
    traindata_cls_counts[i] = load_data(os.path.join(_path, "train/", "task_{}".format(i), "traindata_cls_counts.pkl"))
    testdata_cls_counts[i] = load_data(os.path.join(_path, "test/", "task_{}".format(i), "testdata_cls_counts.pkl"))

train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                  args.datadir,
                                                                                  args.batch_size,
                                                                                  32)

print("len train_ds_global:", len(train_ds_global))
print("len test_ds_global:", len(test_ds_global))

# ########################################## Shared Data
idxs_test = np.arange(len(test_ds_global))
labels_test = np.array(test_ds_global.target)
# Sort Labels Train
idxs_labels_test = np.vstack((idxs_test, labels_test))
idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]

idxs_test = idxs_labels_test[0, :]      # The subscripts in the original array corresponding to the sorted sample labels
labels_test = idxs_labels_test[1, :]    # Sorted sample labels

idxs_test_shared = []
N = args.nsamples_shared//args.nclasses
ind = 0
for k in range(args.nclasses):
    # Return the coordinates of the elements that satisfy the condition labels_test==k in the form of tuples
    ind = max(np.where(labels_test==k)[0])
    idxs_test_shared.extend(idxs_test[(ind - N):(ind)]) # Take the last 10 samples of each label as the shared test data

test_targets = np.array(test_ds_global.target)
shared_data_loader = DataLoader(DatasetSplit(test_ds_global, idxs_test_shared), batch_size=args.nsamples_shared, shuffle=False)

global shared_data
for x, y in shared_data_loader:
    x_flat = x.reshape(len(x), -1)
    x_norm = norm(x_flat)
    print("x_flat.shape is: {}".format(x_norm.shape), flush=True)
    shared_data = TensorDataset(x_norm, y)

# ################################## build model
def init_nets(args, dropout_p=0.5):
    users_model = []

    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16, 8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
            elif args.dataset == 'wm811k':
                # net = SimpleCNNWM(output_dim=8).to(args.device)
                net = SimpleCNNWM2(input_dim=(32 * 9 * 9), hidden_dims=[120, 84], output_dim=8).to(args.device)
        elif args.model == "simple-cnn-3":
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120 * 3, 84 * 3], output_dim=100).to(
                    args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                              output_dim=200).to(args.device)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST().to(args.device)
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN().to(args.device)
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet9':
            if args.dataset == 'cifar100':
                net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'tinyimagenet':
                net = ResNet9(in_channels=3, num_classes=200, dim=512 * 2 * 2)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1:
            net_glob = copy.deepcopy(net)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict, server_state_dict


print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

del server_state_dict
gc.collect()

total = 0
for name, param in net_glob.named_parameters():
    total += np.prod(param.size())
total_size_bytes = total * 4  # Suppose each parameter occupies 4 bytes (a 32-bit floating-point number)
total_size_kb = total_size_bytes / 1024
total_size_mb = total_size_kb / 1024
print(f"Total number of parameters: {total}")
print(f"Model size: {total_size_bytes} bytes")
print(f"Model size: {total_size_kb} KB")
print(f"Model size: {total_size_mb} MB")

clients = []
labels_list = []
counts_list = []

MHV_clients = []
OTDD_clients = []

elapsed_time_otdd = 0
elapsed_time_MH = 0

for idx in range(args.num_users):

    dataidxs = net_dataidx_map[idx]
    if net_dataidx_map_test is None:
        dataidx_test = None
    else:
        dataidxs_test = net_dataidx_map_test[idx]

    print(f'Initializing Client {idx}')

    noise_level = args.noise
    if idx == args.num_users - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                      args.datadir, args.local_bs, 32,
                                                                                      dataidxs, noise_level, idx,
                                                                                      args.num_users - 1,
                                                                                      dataidxs_test=dataidxs_test)
    else:
        noise_level = args.noise / (args.num_users - 1) * idx
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                      args.datadir, args.local_bs, 32,
                                                                                      dataidxs, noise_level,
                                                                                      dataidxs_test=dataidxs_test)
    # client_labels = np.array(train_ds_local.target)
    process = psutil.Process()
    print(f"Current memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

    uni_labels, cnt_labels = list(traindata_cls_counts[idx].keys()), list(traindata_cls_counts[idx].values())
    print(f'Labels: {uni_labels}, Counts: {cnt_labels}')

    # Obtain the TensorDataset instance object of the client dataset
    train_ds_reshape = train_ds_local.data.reshape(len(train_ds_local.data), -1)
    train_ds_temp = norm(train_ds_reshape)
    train_x_local = train_ds_temp
    train_y_local = train_ds_local.target
    if args.dataset in ('cifar10', 'svhn', 'cifar100', 'tinyimagenet'):
        train_x_local = torch.from_numpy(train_ds_temp)
        train_y_local = torch.from_numpy(train_ds_local.target)
    elif args.dataset == 'wm811k':
        train_y_local = torch.from_numpy(train_y_local)
    else:
        pass
    data_local = TensorDataset(train_x_local, train_y_local)

    # Downsampling the client dataset
    local_sample = data_local.subsample(len(shared_data), equal_classes_ratio=True)

    # cifar10_sample = cifar10_data.pca_downsample(args.n_components, args.nsamples_shared)

    start_time_otdd = time.time()
    # Define the optimal transport distance function
    distance_tensorized = PytorchEuclideanDistance()
    routine_tensorized = SinkhornTensorized(distance_tensorized)
    cost_tensorized = SamplesLossTensorized(routine_tensorized)
    print(type(shared_data.features), type(local_sample.features), type(shared_data.labels), type(local_sample.labels))
    outputs = cost_tensorized.distance_with_labels(shared_data.features,
                                                   local_sample.features,
                                                   shared_data.labels,
                                                   local_sample.labels,
                                                   gaussian_class_distance=False)
    OTDD_clients.append(outputs[0].item())
    end_time_otdd = time.time()
    consume_time = end_time_otdd - start_time_otdd
    elapsed_time_otdd += consume_time
    print("Client：{}, the time consumed by otdd in calculating similarity is：{}".format(idx, consume_time))

    labels_list.append(uni_labels)
    counts_list.append(cnt_labels)

    start_time_MH = time.time()
    # The minhash of the labels held by the client
    minhash = MinHash(num_perm=args.num_perm, seed=5386)
    for label in uni_labels:
        minhash.update(str(label).encode('utf-8'))
        # minhash.update(str(client_labels[ii]).encode('utf-8'))
    MHV_clients.append(minhash.hashvalues)

    end_time_MH = time.time()
    consume_time = end_time_MH - start_time_MH
    # print("Client：{}；The time consumed by MinHash is：{}".format(idx, consume_time))
    elapsed_time_MH += consume_time

    # client initialization
    clients.append(Client_ClusterFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                    args.lr, args.momentum, args.device, train_dl_local, test_dl_local,
                                    train_ds_local, test_ds_local))

print("For all clients, The average consumption time of OTDD is：{}".format(elapsed_time_otdd/args.num_users))
print("For all clients, The average consumption time of MinHash is：{}".format(elapsed_time_MH/args.num_users))

# ##################################### Clustering
np.set_printoptions(precision=2)

clusters, u = None, None

if args.use_feature == "otdd":
    print("OTDD-only:")
    arr_OTDD_clients = np.array(OTDD_clients)
    new_OTDD_clients = arr_OTDD_clients[:, np.newaxis]
    new_OTDD_clients_list = list(new_OTDD_clients)
    clusters, u = overlapping_clustering.fcm_clustering(copy.deepcopy(new_OTDD_clients), n_clusters=args.nclusters, m=2)
    print('Clusters: ')
    print(clusters)
    print(f'Number of Clusters {len(clusters)}')
    for jj in range(len(clusters)):
        print(f'Cluster {jj}: {len(clusters[jj])} Users, (clients){clusters[jj]}')
elif args.use_feature == "minhash":
    print("MinHash-only:")
    clusters, u = overlapping_clustering.fcm_clustering(copy.deepcopy(MHV_clients), n_clusters=args.nclusters, m=2)
    print('Clusters: ')
    print(clusters)
    print(f'Number of Clusters {len(clusters)}')
    for jj in range(len(clusters)):
        print(f'Cluster {jj}: {len(clusters[jj])} Users, (clients){clusters[jj]}')
elif args.use_feature == "mix":
    start_time_mixed = time.time()
    print("MinHash+OTDD:")
    mix_signatures = [list(item1)+[item2] for item1, item2 in zip(MHV_clients, OTDD_clients)]
    print("MinHash signatures are: {}".format(mix_signatures))

    # clusters = hierarchical_clustering_mix(copy.deepcopy(mix_signatures), distance_fn=custom_distance, num_clusters=args.nclusters)
    # clusters, u = overlapping_clustering.fcm_clustering(copy.deepcopy(mix_signatures), n_clusters=args.nclusters, m=2)
    clusters, u = overlapping_clustering.fcm_revise(copy.deepcopy(mix_signatures), n_clusters=args.nclusters,
                                                            custom_distance=custom_distance, num_vote=args.num)
    print('Clusters: ')
    print(clusters)
    print('')
    print(f'Number of Clusters {len(clusters)}')
    print('')

    end_time_mixed = time.time()
    elapsed_time_mixed = end_time_mixed - start_time_mixed
    print("The judgment time for cluster identity of clients is：{}".format(elapsed_time_mixed))

    for jj in range(len(clusters)):
        print(f'Cluster {jj}: {len(clusters[jj])} Users, (clients){clusters[jj]}')
else:
    print("not supported yet")
    exit(1)

# Clustering Error
idxs_users = np.arange(0, args.num_users, 1)
clust_err, clust_acc = error_clustering(clusters, idxs_users, clients, traindata_cls_counts)
print("The clustering error is {};\tThe clustering accuracy is {}.\n".format(clust_err, clust_acc))

# ##################################### Federation
loss_train = []
loss_locals = []
init_local_tacc = []  # initial local test accuracy at each round
final_local_tacc = []  # final local test accuracy at each round
init_local_tloss = []  # initial local test loss at each round
final_local_tloss = []  # final local test loss at each round
init_tacc_pr = []  # initial test accuarcy for each round
final_tacc_pr = []  # final test accuracy for each round
init_tloss_pr = []  # initial test loss for each round
final_tloss_pr = []  # final test loss for each round
clients_best_acc = [0 for _ in range(args.num_users)]
ckp_avg_tacc = []
ckp_avg_best_tacc = []
cur_tacc_glob = {k: 0 for k in range(args.num_users)}   # global accuracy for each round (global models)
cur_tacc_local = {k: 0 for k in range(args.num_users)}  # local accuracy for each round (client models)
tacc_per_round_global = []     # average global accuracy for all round
tacc_per_round_local = []     # average local accuracy for all round
w_glob_per_cluster = [copy.deepcopy(initial_state_dict) for _ in range(len(clusters))]
best_glob_acc = [0 for _ in range(len(clusters))]

for idnx in range(args.num_users):
    _, acc = clients[idnx].eval_test()
    cur_tacc_local[idnx] = acc

    idx_clusters = [i for i, row in enumerate(clusters) if idnx in row]
    w_glob_users = []  # The set of cluster models corresponding to client idnx
    wgts = []
    if len(idx_clusters) > 1:
        for clust in idx_clusters:  # u[cluster][client]
            w_glob_users.append(copy.deepcopy(w_glob_per_cluster[clust]))
            wgts.append(u[clust][idnx])
        user_glob_w = FedAvg(w_glob_users, weight_avg=wgts)  # Send the average model of cluster models correspinding to client idnx
    else:
        user_glob_w = w_glob_per_cluster[idx_clusters[0]]
    clients[idnx].set_state_dict(copy.deepcopy(user_glob_w))
    _, acc = clients[idnx].eval_test()
    cur_tacc_glob[idnx] = acc
    torch.cuda.empty_cache()

# ======================= Training ===================
print_flag = False
n_users = args.num_users
consume_time_all_round = 0
consume_time_all_client = 0
total_n_client = 0
for iteration in range(args.rounds):
    start_time_once = time.time()

    m = max(int(args.frac * n_users), 1)
    idxs_users = np.random.choice(range(n_users), m, replace=False)
    total_n_client += m

    print(f'###### ROUND {iteration + 1} ######')
    print(f'Clients {idxs_users}')

    idx_clusters_round = {clust: [] for clust in range(len(clusters))}
    for idx in idxs_users:
        start_time = time.time()
        # Obtain the list of cluster identity to which the client idx belongs
        idx_clusters = [i for i, row in enumerate(clusters) if idx in row]
        w_glob_users = []  # The set of cluster models corresponding to client idx
        wgts = []        # with affiliation degree
        if len(idx_clusters) > 1:
            for clust in idx_clusters:
                idx_clusters_round[clust].append(idx)
                w_glob_users.append(copy.deepcopy(w_glob_per_cluster[clust]))
                wgts.append(u[clust][idx])        # with affiliation degree
            # Send the weighted average model of cluster models correspinding to client idx
            user_glob_w = FedAvg(w_glob_users, weight_avg=wgts)
            # user_glob_w = FedAvg(w_glob_users)      # without affiliation degree
        else:
            idx_clusters_round[idx_clusters[0]].append(idx)
            user_glob_w = w_glob_per_cluster[idx_clusters[0]]
        clients[idx].set_state_dict(copy.deepcopy(user_glob_w))
        loss, acc = clients[idx].eval_test()
        init_local_tacc.append(acc)
        init_local_tloss.append(loss)
        cur_tacc_glob[idx] = acc

        loss = clients[idx].train(args.alg, None)
        loss_locals.append(copy.deepcopy(loss))
        loss, acc = clients[idx].eval_test()
        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc
        final_local_tacc.append(acc)
        final_local_tloss.append(loss)
        cur_tacc_local[idx] = acc

        end_time = time.time()
        elapsed_time = end_time - start_time
        consume_time_all_client += elapsed_time
        print("Client：{}； training time：{}".format(idx, elapsed_time))

    # ############## Model Aggregation ###############
    total_data_points = {}
    for k in idx_clusters_round.keys():
        temp_sum = []
        for r in idx_clusters_round[k]:
            temp_sum.append(len(net_dataidx_map[r]))        # net_dataidx_map:{client id: the sample id set}
        total_data_points[k] = sum(temp_sum)        # the total number of samples in each cluster

    freqs_data = {}
    freqs_cluster = {}      # with affiliation degree
    for k in idx_clusters_round.keys():
        freqs_data[k] = []
        freqs_cluster[k] = []
        for r in idx_clusters_round[k]:
            ratio = len(net_dataidx_map[r]) / total_data_points[k]  # The proportion of client data size
            freqs_data[k].append(copy.deepcopy(ratio))
            freqs_cluster[k].append(u[k][r])      # The affiliation degree of client to cluster

    # Fusion weight calculation
    fed_avg_freqs = {}
    for k in idx_clusters_round.keys():     # idx_clusters_round  {clust: clients}
        fed_avg_freqs[k] = []
        freq_fusion = [x * y for x, y in zip(freqs_data[k], freqs_cluster[k])]
        for p in freq_fusion:
            ratio = p / sum(freq_fusion)
            fed_avg_freqs[k].append(copy.deepcopy(ratio))

        # Weighted average
        w_locals = []
        for el in idx_clusters_round[k]:
            # w_locals: client models' parameters within the same cluster
            w_locals.append(copy.deepcopy(clients[el].get_state_dict()))
        if len(w_locals) != 0:
            ww = FedAvg(w_locals, weight_avg=fed_avg_freqs[k])      # with affiliation degree
            # ww = FedAvg(w_locals, weight_avg=freqs_data[k])         # without affiliation degree
            w_glob_per_cluster[k] = copy.deepcopy(ww)
            net_glob.load_state_dict(copy.deepcopy(ww))
            _, acc = eval_test(net_glob, args, shared_data_loader)
            if acc > best_glob_acc[k]:
                best_glob_acc[k] = acc
        else:
            pass

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
    mean_g = sum(cur_tacc_glob.values()) / len(cur_tacc_glob)
    mean_l = sum(cur_tacc_local.values()) / len(cur_tacc_local)

    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))
    template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
    print(template.format(avg_init_tloss, avg_init_tacc))
    template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
    print(template.format(avg_final_tloss, avg_final_tacc))
    print("The average test accuracy of the global model on all client data: {}".format(mean_g))
    print("The average test accuracy of the local models: {}".format(mean_l))

    # Print the overall performance periodically
    print_flag = False
    if iteration < 60:
        print_flag = False
    elif iteration % args.print_freq == 0:
        print_flag = True

    if print_flag:
        print('--- PRINTING ALL CLIENTS STATUS ---')
        current_acc = []
        for k in range(n_users):
            loss, acc = clients[k].eval_test()
            current_acc.append(acc)

            template = ("Client {:3d}, best_acc {:3.3f}, current_acc {:3.3f}")
            print(template.format(k, clients_best_acc[k], current_acc[-1]))

        template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
        print(template.format(iteration + 1, np.mean(current_acc), np.mean(clients_best_acc[0: n_users])))

        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc[0: n_users]))


    print('')
    print(f'Clusters {idx_clusters_round}')
    print('')

    loss_train.append(loss_avg)
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)
    tacc_per_round_global.append(mean_g)
    tacc_per_round_local.append(mean_l)

    # break;
    ## clear the placeholders for the next round
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()

    ## calling garbage collector
    gc.collect()

    end_time_once = time.time()
    elapsed_time_once = end_time_once - start_time_once
    consume_time_all_round += elapsed_time_once
    print("Round：{}； running time for each round：{}".format(iteration+1, elapsed_time_once))

print("(Client) Average training time：{}".format(consume_time_all_client/total_n_client))
print("Average training time per round：{}".format(consume_time_all_round/args.rounds))

# ############################## Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

for idx in range(n_users):
    loss, acc = clients[idx].eval_test()

    test_loss.append(loss)
    test_acc.append(acc)

    loss, acc = clients[idx].eval_train()

    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

print(f'Best Clients AVG Acc: {np.mean(clients_best_acc[0: n_users])}')

for jj in range(len(clusters)):
    print(f'Cluster {jj}, Best Glob Acc {best_glob_acc[jj]:.3f}')

print(f'Average Best Glob Acc {np.mean(best_glob_acc[0:len(clusters)]):.3f}')

############################# Saving Print Results
with open(path + 'final_results.txt', 'a') as text_file:
    print(f'Train Loss: {train_loss}, Test_loss: {test_loss}', file=text_file)
    print(f'Train Acc: {train_acc}, Test Acc: {test_acc}', file=text_file)

    print(f'Best Clients AVG Acc: {np.mean(clients_best_acc[0: n_users])}', file=text_file)
    print(f'Best Global Models AVG Acc: {np.mean(best_glob_acc[0:len(clusters)])}', file=text_file)


# print('--------------UnSeen Clients-------------')
# test_loss = []
# test_acc = []
# train_loss = []
# train_acc = []
#
# start_time = time.time()
# for idx in range(n_users, 100):
#     idx_clusters = [i for i, row in enumerate(clusters) if idx in row]
#     # idx_cluster = idx_clusters[idx]
#     for clust in idx_clusters:
#         clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[clust]))
#
#         loss = clients[idx].train(is_print=False)
#         loss_tr, acc_tr = clients[idx].eval_train()
#         train_loss.append(loss_tr)
#         train_acc.append(acc_tr)
#
#         loss_te, acc_te = clients[idx].eval_test()
#         test_loss.append(loss_te)
#         test_acc.append(acc_te)
#
#         if acc_te > clients_best_acc[idx]:
#             clients_best_acc[idx] = acc_te
#
#     template = ("UnSeen Client {:3d}, best_acc {:3.3f}, current_acc {:3.3f} \n")
#     print(template.format(idx, clients_best_acc[idx], test_acc[-1]))
#
# end_time = time.time()
# elapsed_time_cluster = end_time - start_time
# print("The calculation time of fedfoc for new clients：{}".format(elapsed_time_cluster))
#
# template = ("UnSeen Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
# print(template.format(np.mean(test_acc), np.mean(clients_best_acc[n_users:])))
#
# test_loss = sum(test_loss) / len(test_loss)
# test_acc = sum(test_acc) / len(test_acc)
#
# train_loss = sum(train_loss) / len(train_loss)
# train_acc = sum(train_acc) / len(train_acc)
#
# print(f'UnSeen Train Loss: {train_loss}, Test_loss: {test_loss}')
# print(f'UnSeen Train Acc: {train_acc}, Test Acc: {test_acc}')
# print(f'Best ALL Clients AVG Acc: {np.mean(clients_best_acc)}')
