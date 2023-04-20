from numpy import random

import numpy as np

def uniform_distribute(dataset, args):
    globally_shared_data_idx = []
    
    idxs = np.arange(len(dataset))

    
    if args.dataset == "mnist":
        labels = dataset.targets

    elif args.dataset == "cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    
    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)

        globally_shared_data = np.random.choice(specific_class, int(args.alpha * args.classwise), replace=False)

        globally_shared_data_idx = globally_shared_data_idx + list(globally_shared_data)

    return globally_shared_data_idx


def uniform_distribute1(dataset, args):
    idxs = np.arange(len(dataset))
    # print(idxs)

    if args.dataset == "mnist":
        labels = dataset.targets

    elif args.dataset == "cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')

    random.shuffle(idxs)
    random.shuffle(idxs)

    return idxs

def train_dg_split(dataset, args): 
    dg_idx = []
    train_idx = []
    idxs = np.arange(len(dataset))

    if args.dataset == "mnist":
        labels = dataset.targets
    else:
        exit('Error: unrecognized dataset')

    idxs_labels = np.vstack((idxs, labels))  #将两个数组在竖直方向进行堆叠并获得新的数组

    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] #按标签值大小排列后输出对应索引值，上下一起按照索引值变换
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    args.num_classes=7
    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)

        dg = np.random.choice(specific_class, args.classwise)


        train_tmp = set(specific_class)-set(dg)

        
        dg_idx = dg_idx + list(dg)
        
        train_idx = train_idx + list(train_tmp)

    return dg_idx, train_idx    