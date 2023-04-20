import copy
import math

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from opacus import PrivacyEngine
import numpy as np
from random import randint
from opacus import PrivacyEngine
from src.test import tes_img
from utils.dp_mechanism import cal_sensitivity_up, Laplace, Gaussian_Simple, Gaussian_moment


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class ModelUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, dp_epsilon=10, dp_delta=1e-5, dp_clip=5,max_train_data_len=None ,min_train_data_len = None,dataset_all_len = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # args.local_bs=10，shuffle=True表示每次迭代训练时将数据洗牌
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_train = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        self.idxs = idxs
        self.len_dataset = len(dataset)
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.max_train_data_len = max_train_data_len
        self.min_train_data_len = min_train_data_len
        # self.dp_clip = dp_clip*(len(dataset)/dataset_all_len)
        # self.dp_clip = dp_clip * (min_train_data_len / dataset_all_len)
        self.dp_clip = dp_clip*(max_train_data_len/dataset_all_len)
        # self.dp_clip = dp_clip/len(dataset)
        # self.dp_clip = dp_clip/self.min_train_data_len
        # self.dp_clip = dp_clip/self.max_train_data_len
        # self.dp_clip = dp_clip
    def train(self, local_net, net):

        net.train()
        # print(self.dp_clip)
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

        epoch_loss = []

        if self.args.sys_homo:
            local_ep = self.args.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le)

        #生成空的參數框架,用於存放梯度值
        # grad_net = copy.deepcopy(net.state_dict())
        # for k in grad_net.keys():
        #     grad_net[k] = torch.zeros_like(grad_net[k], dtype=torch.float32).to(self.args.device)  #清空

        start_net = copy.deepcopy(net)
        for iter in range(local_ep):
            batch_loss = []
            grad_all = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()  # 清空梯度
                log_probs = net(images)
                # print(log_probs)
                loss = self.loss_func(log_probs, labels)
                # print(loss)
                loss.backward()  # 求梯度
                # self.clip_gradients(net)  #梯度裁減

                optimizer.step()  # 更新参数
                scheduler.step()  # 调整学习率

                batch_loss.append(loss.item())

                # for k,v in net.named_parameters():
                #     grad_net[k] += (v.grad/math.ceil(self.len_dataset/self.args.local_bs))
                # print(math.ceil(self.len_dataset/self.args.local_bs))

                #查看
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #               100. * batch_idx / len(self.ldr_train), loss.item()))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        # self.clip_gradients(net)  #梯度裁減
        # with torch.no_grad():
        #     for k, v in net.named_parameters():
        #         v -=  self.args.lr*grad_net[k]
        # 更新前后參數差值
        delta_net = copy.deepcopy(net)
        # 參數裁剪
        with torch.no_grad():
            for k, v in delta_net.named_parameters():
                v -= start_net.state_dict()[k]
        # with torch.no_grad():
        #     self.clip_parameters(delta_net)
        with torch.no_grad():
            for k, v in start_net.named_parameters():
                v += delta_net.state_dict()[k]

        # 给参数加噪
        # print(self.len_dataset)
        # print(self.dp_clip)
        net_noise = copy.deepcopy(start_net)
        # self.add_noise(net_noise)
        return start_net.state_dict(),net_noise.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def clip_gradients(self, net):

        # Laplace use 1 norm
        # for k, v in net.named_parameters():
        #     v.grad /= max(1, v.grad.norm(1) / self.dp_clip)

        # Gaussian use 2 norm
        for k, v in net.named_parameters():
            # if v.grad.norm(2) > 2:
           print(v.grad.norm(2).item())
           v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
           # print(v.grad.norm(2))

    def clip_parameters(self, net):
        for k, v in net.named_parameters():
            # if v.norm(2).item() > self.dp_clip:
            #    print(v.norm(2).item())
            v /= max(1, v.norm(2).item() / self.dp_clip)
            # print(v.norm(2))
    def add_noise(self, net):
        sensitivity = cal_sensitivity_up(self.args.lr, self.dp_clip)
        # 拉普拉斯
        # with torch.no_grad():
        #     for k, v in net.named_parameters():
        #         noise = Laplace(epsilon=5, sensitivity=sensitivity, size=v.shape)
        #         noise = torch.from_numpy(noise).to(self.args.device)
        #         v += noise
        #  高斯
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(self.args.device)
                v += noise
