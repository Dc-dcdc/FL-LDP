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

        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

        self.idxs = idxs
        self.len_dataset = len(dataset)
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.max_train_data_len = max_train_data_len
        self.dp_clip = dp_clip*(max_train_data_len/dataset_all_len)

    def train(self, local_net, net):

        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

        epoch_loss = []

        if self.args.sys_homo:
            local_ep = self.args.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le)



        start_net = copy.deepcopy(net)
        for iter in range(local_ep):
            batch_loss = []
            grad_all = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                loss.backward()


                optimizer.step()
                scheduler.step()

                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        delta_net = copy.deepcopy(net)

        with torch.no_grad():
            for k, v in delta_net.named_parameters():
                v -= start_net.state_dict()[k]
        # with torch.no_grad():
        #     self.clip_parameters(delta_net)
        with torch.no_grad():
            for k, v in start_net.named_parameters():
                v += delta_net.state_dict()[k]

        net_noise = copy.deepcopy(start_net)
        # self.add_noise(net_noise)
        return start_net.state_dict(),net_noise.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def clip_parameters(self, net):
        for k, v in net.named_parameters():
            v /= max(1, v.norm(2).item() / self.dp_clip)

    def add_noise(self, net):
        sensitivity = cal_sensitivity_up(self.args.lr, self.dp_clip)
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(self.args.device)
                v += noise
