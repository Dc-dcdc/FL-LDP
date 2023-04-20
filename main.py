import math
from datetime import datetime

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
from tqdm import trange

from utils.distribute import uniform_distribute, train_dg_split, uniform_distribute1
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import  CNN_v1, CNN_v2, CNN_v3, CNN_v4
from src.strategy import FedAvg
from src.test import tes_img
import pandas as pd
import csv
writer = SummaryWriter()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
 
    random.seed(args.seed)    #会改变随机生成器的种子,使得每次随机输出都是一样的
    np.random.seed(args.seed)   #按顺序产生一组固定的数组;
    torch.manual_seed(args.seed)  #为cpu设置随机数种子
    torch.cuda.manual_seed_all(args.seed)  #为所有GPU设置随机数种子

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # 转为tensor型并进行归一化，因为只有一层所以只需要填第一个就行，output=（input-0.5）/0.5  (0,1)--(-1,1)
        dataset = torchvision.datasets.ImageFolder('Z:/FL/代码/师兄源代码/6x6数据/train',
														 transform=trans_mnist)
        dataset_test={i: list() for i in range(0,7)}
        for i in range(0,7):
             dataset_test[i] = torchvision.datasets.ImageFolder(('Z:/FL/代码/师兄源代码/6x6数据/test/{}'.format(i)),
														transform=trans_mnist)

        dataset_share1 = torchvision.datasets.ImageFolder('Z:/FL/代码/师兄源代码/6x6数据/share_data1',
                                                        transform=trans_mnist)
        dataset_share2 = torchvision.datasets.ImageFolder('Z:/FL/代码/师兄源代码/6x6数据/share_data2',
                                                        transform=trans_mnist)
        # dataset = datasets.MNIST('../data/mnist', train=True, download=True, transform=trans_mnist)
        # dataset_test = datasets.MNIST('../data/mnist', train=True, download=True, transform=trans_mnist)

        dataset_all_len = len(dataset)   #数据集数量
        train_data = {i: list() for i in range(args.num_users)}
        for i in range(0,args.num_users):
            train_data[i] = copy.deepcopy(dataset)

        dataset_train = copy.deepcopy(dataset)

        if args.sampling == 'iid':
            local_dix = iid(dataset_train, args.num_users)
        elif args.sampling == 'noniid':
            local_dix = noniid(dataset_train, args)
        else:
            exit('Error: unrecognized sampling')
    else:
        exit('Error: unrecognized dataset')
    
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn2' and args.dataset == 'mnist':
        print("alexnet")
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'cnn3'and args.dataset == 'mnist':
        print("resnet18")
        net_glob = CNN_v3(args=args).to(args.device)
    elif args.model == 'cnn4' and args.dataset == 'mnist':
        print("vgg16")
        net_glob = CNN_v4(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)

    # distribute globally shared data (uniform distribution)
    # share_idx = uniform_distribute(dg, args)
    share_idx = uniform_distribute1(dataset_share1, args)
    share2_idx = uniform_distribute1(dataset_share2, args)

    idxs_users = range(args.num_users)
    train_data_len = []
    for idx in idxs_users:
        ##计算共享数据数量，第一步
        #用于测试共享数据率
        # share_idx1 = np.round(np.linspace(0, len(share2_idx) - 1,490)).astype(int)

        #用于ideal情况
        # share_idx1 = np.round(np.linspace(0, len(share_idx) - 1,int((1 - (len(set(list(local_dix[idx]))) / 7000)) * len(share_idx)))).astype(int)
        # 用于Non-ideal情况，全部下发，或者用于对比实验
        share_idx1 = share_idx
        #第二步取共享数据集
        if len(set(list(local_dix[idx])))<len(dataset)/5 :
           share2_idx1 = np.round(np.linspace(0, len(share2_idx) - 1,int(len(dataset)/5)-len(set(list(local_dix[idx]))))).astype(int)
        else :
           share2_idx1 = []
        # 取出共享数据中对应的数据序列
        share1_idx2 = res = list(map(share_idx.__getitem__, share_idx1))

        # 测试共享数据率
        # share2_idx2 = list(map(share2_idx.__getitem__, share_idx1))

        #取出第二次
        share2_idx2 = list(map(share2_idx.__getitem__, share2_idx1))
        train_data[idx].imgs = []
        train_data[idx].targets = []
        train_data[idx].samples = []
        for i in local_dix[idx]:   #本地数据
            train_data[idx].imgs.append(dataset.imgs[i])
            train_data[idx].targets.append(dataset.targets[i])
            train_data[idx].samples.append(dataset.samples[i])
        # for j in share1_idx2:   #第一步共享数据
        #     train_data[idx].imgs.append(dataset_share1.imgs[j])
        #     train_data[idx].targets.append(dataset_share1.targets[j])
        #     train_data[idx].samples.append(dataset_share1.samples[j])

        # #测试共享数据量
        # for j in share2_idx2:   #
        #     train_data[idx].imgs.append(dataset_share2.imgs[j])
        #     train_data[idx].targets.append(dataset_share2.targets[j])
        #     train_data[idx].samples.append(dataset_share2.samples[j])


        # for k in share2_idx2:  #第二步共享数据
        #     train_data[idx].imgs.append(dataset_share2.imgs[k])
        #     train_data[idx].targets.append(dataset_share2.targets[k])
        #     train_data[idx].samples.append(dataset_share2.samples[k])
        train_data_len.append(len(train_data[idx]))
    max_train_data_len = max(train_data_len)
    min_train_data_len = min(train_data_len)
    print('max:',max_train_data_len)
    print('min:',min_train_data_len)
    #初始参数生成
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # initialization stage of FedShare
    dg_idx = np.arange(len(dataset))
    initialization_stage = ModelUpdate(args=args, dataset=dataset, idxs=set(dg_idx),max_train_data_len=max_train_data_len,min_train_data_len = min_train_data_len,dataset_all_len =dataset_all_len)
    w_glob, _,_ = initialization_stage.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_glob)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        w_locals_noise = [w_glob for i in range(args.num_users)]

    for iter in trange(args.rounds):

        if not args.all_clients:
            w_locals = []
            w_locals_noise =[]
        users_loss = 0
        users_loss_avg = 0
        # m = max(int(args.frac * args.num_users), 1)
        #
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # Local update
            # 本地训练数据量为 本地数据+共享数据
            local = ModelUpdate(args=args, dataset=train_data[idx], idxs=local_dix[idx], max_train_data_len=max_train_data_len,min_train_data_len = min_train_data_len,dataset_all_len = dataset_all_len)
            # print(idxs_users)
            # local = ModelUpdate(args=args, dataset=dataset, idxs=set(list(dict_users[idx]) + share_idx))
            w,w_noise, loss = local.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
            users_loss += len(train_data[idx])/len(dataset)*loss #权重
            users_loss_avg += 1/args.num_users*loss              #平均

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            if args.all_clients:
                w_locals_noise[idx] = copy.deepcopy(w_noise)
            else:
                w_locals_noise.append(copy.deepcopy(w_noise))

                # update global weights
            w_glob = FedAvg(w_locals, args)  #without noise
            w_glob_noise = FedAvg(w_locals_noise, args)   #with noise
            #信噪比计算
            SNR_layers = []
            state_dict = copy.deepcopy(w_glob)
            mean = lambda x: sum(x) / len(x)
            for key in state_dict:
                SNR_user = []
                for i in idxs_users:
                    # a = torch.var(w_locals[i][key] - w_glob[key]) / torch.var(w_noise[key] - w_locals[i][key])
                    # b = a[0]
                    SNR_user.append(
                        (torch.var(w_locals[i][key] - w_glob[key]) / torch.var(w_locals_noise[i][key] - w_locals[i][key])).item())
                SNR_layers.append(mean(SNR_user))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob_noise)
        # acc_test = [0,1,2,3,4,5,6]
        # loss_test = {i: list() for i in range(0,7)}
        # for  i in range(0,7):
        correct_every_fault = [[0 for i in range(7)] for j in range(7)]
        acc_test, loss_test, correct_every_fault= tes_img(net_glob, dataset_test, args)

        if args.debug:
             print(f"Round: {iter}")
             print(f"Test accuracy: {acc_test}")
             if args.title == 1:
                args.title = 0
                header = {'Iteration':[], 'acc_test':[], 'loss_test':[], 'idx_loss_avg':[], 'idx_loss_w':[], 'SNR':[],'hour':[], 'min':[]}
                # header = ['Iteration', 'acc_test', 'loss_test', 'idx_loss_avg', 'idx_loss',
                #           'hour', 'min']
                data = pd.DataFrame(header)
                data.to_csv('Z:/FL/代码/师兄源代码/6x6数据/实验数据.csv', mode='w', index=False)  # mod

             dt = datetime.now()
             lists = [iter, round(acc_test,2), round(loss_test,3),round(users_loss_avg,4),round(users_loss,4),round(-10*math.log10(mean(SNR_layers)),3),dt.hour,dt]
             for i in range(0,7):
                 for j in range(0, 7):
                     lists.append(correct_every_fault[i][j])
             data = pd.DataFrame([lists])

             data.to_csv('Z:/FL/代码/师兄源代码/6x6数据/实验数据.csv', mode='a', header=False,
                        index=False)  # mod
             # print(f"Test loss: {loss_test}")
             # print(f'TIME：{dt.year}Y{dt.month}M{dt.day}D {dt.hour}:{dt.minute}:{dt.second}')
        
        # tensorboard
        if args.tsboard:
            writer.add_scalar(f"Test accuracy:Share{args.dataset}, {args.fed}", acc_test, iter)
            writer.add_scalar(f"Test loss:Share{args.dataset}, {args.fed}", loss_test, iter)

    writer.close()