import argparse


## CIFAR-10 has 50000 training images (5000 per class), 10 classes, 10000 test images (1000 per class)
## CIFAR-100 has 50000 training images (500 per class), 100 classes, 10000 test images (100 per class)
## MNIST has 60000 training images (min: 5421, max: 6742 per class), 10000 test images (min: 892, max: 1135
## per class) --> in the code we fixed 5000 training image per class, and 900 test image per class to be
## consistent with CIFAR-10

## CIFAR-10 Non-IID 250 samples per label for 2 class non-iid is the benchmark (500 samples for each client)

def args_parser():
    parser = argparse.ArgumentParser()
    # dataset preparing
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="name of dataset: cifar10, cifar100, fmnist, svhn, wm811k")
    parser.add_argument('--nclasses', type=int, default=10, help="number of classes")
    parser.add_argument('--nsamples_shared', type=int, default=100, help="number of shared data samples")

    # dataset partitioning arguments
    parser.add_argument('--partition', type=str, default='Dir', help='method of partitioning: Dir, noniid2, noniid3')
    parser.add_argument('--beta', type=float, default=0.1, help='The parameter for the dirichlet distribution for data' \
                                                                'partitioning: 0.1 or 0.5')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')

    # model arguments
    parser.add_argument('--model', type=str, default='simple-cnn', help='model name')

    # federated arguments
    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--batch_size', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--trial', type=int, default=1, help="the trial number")
    parser.add_argument('--alg', type=str, default='fedfoc', help='Algorithm')
    parser.add_argument('--datadir', type=str, default='./data/data', help='data directory')
    parser.add_argument('--savedir', type=str, default='./save_results/', help='save directory')
    parser.add_argument('--logdir', type=str, default='./logs/', help='logs directory')
    parser.add_argument('--local_view', action='store_true', help='whether from client perspective or not')

    # clustering arguments
    parser.add_argument('--num', type=int, default=2, help="The number of vote")
    parser.add_argument('--nclusters', type=int, default=3, help="Number of Clusters for FedFOC")
    parser.add_argument('--linkage', type=str, default="average", help="The linkage of AHC")

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--print_freq', type=int, default=10, help="printing frequency during training rounds")
    parser.add_argument('--num_perm', type=int, default=4, help='define the number of hash functions')
    parser.add_argument('--weight_distance', type=float, default=0.5, help='weight of Jaccard distance')
    parser.add_argument('--use_feature', type=str, default='mix', help='features for clustering')

    parser.add_argument('--seed', type=int, default=2025, help='random seed (default: 1)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')

    args = parser.parse_args()
    return args
