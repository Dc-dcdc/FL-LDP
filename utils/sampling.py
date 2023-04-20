import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):

        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

def noniid(dataset, args):

    num_dataset = len(dataset)

    idx = np.arange(num_dataset)

    dict_users = {i: list() for i in range(args.num_users)}


    dict_users[0] = idx[0:1505]    #1505
    dict_users[1] = idx[1505:2612] #1107
    dict_users[2] = idx[2612:4462] #1850
    dict_users[3] = idx[4462:5817] #1355
    dict_users[4] = idx[5817:7000] #1183


    # dict_users[0] = idx[0:220]    #220
    # dict_users[1] = idx[220:1874] #1652
    # dict_users[2] = idx[1874:2460] #586
    # dict_users[3] = idx[2460:5498] #3038
    # dict_users[4] = idx[5498:7000] #1502

    print(dict_users)
    return dict_users