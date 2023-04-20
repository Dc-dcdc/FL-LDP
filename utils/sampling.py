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
    # print(num_dataset)
    idx = np.arange(num_dataset)

    # #用于打乱以满足IID
    # for i in np.arange(len(idx) - 1, 0, -1):
    #     idx1 = np.random.choice(range(i))
    #     idx[i], idx[idx1] = idx[idx1], idx[i]
    # print(idx)
    dict_users = {i: list() for i in range(args.num_users)}
    # print(dict_users)

    # min_num = 1
    # max_num = 4300

    # random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    # print(f"Total number of datasets owned by clients : {sum(random_num_size)}")
    # #
    # # # total dataset should be larger or equal to sum of splitted dataset.
    # #
    # print(sum(random_num_size))
    # assert num_dataset >= sum(random_num_size)

    # divide and assign
    #ideal情况
    dict_users[0] = idx[0:1505]    #1505
    dict_users[1] = idx[1505:2612] #1107
    dict_users[2] = idx[2612:4462] #1850
    dict_users[3] = idx[4462:5817] #1355
    dict_users[4] = idx[5817:7000] #1183
    #non-ideal情况
    # dict_users[0] = idx[0:217]    #2205
    # dict_users[1] = idx[217:799] #1407
    # dict_users[2] = idx[799:1164] #3150
    # dict_users[3] = idx[1164:1912] #1155
    # dict_users[4] = idx[1912:10500] #2583

    # dict_users[0] = idx[0:220]    #220
    # dict_users[1] = idx[220:1874] #1652
    # dict_users[2] = idx[1874:2460] #586
    # dict_users[3] = idx[2460:5498] #3038
    # dict_users[4] = idx[5498:7000] #1502

    # dict_users[0] = idx[0:20]    #20
    # dict_users[1] = idx[20:1074] #1052
    # dict_users[2] = idx[1074:1460] #486
    # dict_users[3] = idx[1460:6498] #5038
    # dict_users[4] = idx[5038:7000] #962

    print(dict_users)
    return dict_users