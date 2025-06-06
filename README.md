# FedFOC: Personalized Federated Learning via Fine-grained One-shot Clustering (FedFOC) **[Accepted at TII 2025]**

In this repository, we release the official code of FedFOC algorithm.


## Usage

Build and Activate a virtual environment
```
conda env create -f environment.yml
conda activate fedfoc
```

Generate distributed datasets **[The default parameters in options_cluster.py can be modified as needed]**
```
cd FedFOC
python generate_data.py
```

For WM-811K datase: [MIR-WM811K](http://mirlab.org/dataSet/public/), Run the following command to process it
```
python wm811k-process.py
```

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
bash scripts/fedfoc.sh
```
Please follow the paper to modify the scripts for more experiments. You may change the parameters listed in the following table.

The descriptions of parameters are as follows:
| Parameter | Description |  
| --------- | ----------- |  
| rounds            | The number of communication rounds per run. |  
| num_users         | The number of clients. |  
| frac              | The sampling rate of clients for each round. |  
| local_ep          | The number of local training epochs. |  
| local_bs          | Local batch size. |  
| lr                | The learning rate for local models. |  
| momentum          | The momentum for the optimizer. |
| model             | Network architecture. Options: simple-cnn, resnet9 |  
| dataset           | The dataset for training and testing. Options are discussed in options_cluster.py. |  
| datadir           | The path of datasets. |  
| logdir            | The path to store logs. |  
| savedir           | The path to store results. |  
| partition         | How datasets are partitioned. Options: `homo`, `Dir`, `noniid2` (or 3, ..., which means the fixed number of labels each user owns). |  
| beta              | The concentration parameter of the Dirichlet distribution for heterogeneous partition. Options: 0.1, 0.5|  
| alg               | Federated learning algorithm. Options are discussed above. |  
| local_view        | If true puts local test set for each client |  
| nclusters         | The number of clusters |  
| nclasses          | The number of classes of datasets |  
| nsamples_shared   | The number of samples in auxiliary dataset |  
| print_freq        | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |  
| num_perm          | The number of MinHash signatures |  
| num               | The number of voting threshold |  
| weight_distance   | The weight of Jaccard distance |  
| gpu               | The IDs of GPU to use. E.g., 0 |  

## Citation 
Please cite our work if you find it relavent to your research and used our implementations.

## Acknowledgements
Some parts of our code and implementation has been adapted from [PACFL](https://github.com/MMorafah/PACFL) repository.
