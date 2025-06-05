from src.utils import *
import pickle

args = args_parser()

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath, exist_ok=True)
    except Exception as _:
        pass


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)

# #################################### Data partitioning section
args.local_view = True
X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset,
                                                               args.datadir, args.logdir, args.partition,
                                                               args.num_users, beta=args.beta,
                                                               local_view=args.local_view)

print("当前路径为：{}".format(os.getcwd()))
# 保存数据
for i in net_dataidx_map.keys():
    if args.partition == "Dir":
        train_path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition+str(args.beta),
                                  "train", "task_{}".format(i))
    else:
        train_path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition, "train", "task_{}".format(i))
    # print("train_path:", train_path)
    mkdirs(train_path)
    save_data(net_dataidx_map[i], os.path.join(train_path, "train.pkl"))
    save_data(traindata_cls_counts[i], os.path.join(train_path, "traindata_cls_counts.pkl"))

    # print("net_dataidx_map len:{}\n".format(len(net_dataidx_map[i])))
for j in net_dataidx_map_test.keys():
    if args.partition == "Dir":
        test_path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition+str(args.beta), "test", "task_{}".format(j))
    else:
        test_path = os.path.join(os.getcwd(), "data/partitioned", args.dataset, args.partition, "test", "task_{}".format(j))
    mkdirs(test_path)
    save_data(net_dataidx_map_test[j], os.path.join(test_path, "test.pkl"))
    save_data(testdata_cls_counts[j], os.path.join(test_path, "testdata_cls_counts.pkl"))

    # print("net_dataidx_map_test len:{}\n".format(len(net_dataidx_map_test[j])))