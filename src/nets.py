import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# 编写Inception模块
# class Inception(nn.Module):
#     def __init__(self, in_planes,
#                  n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
#         super(Inception, self).__init__()
#         # 1x1 conv branch
#         self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)
#
#         # 1x1 conv -> 3x3 conv branch
#         self.b2_1x1_a = BasicConv2d(in_planes, n3x3red,
#                                     kernel_size=1)
#         self.b2_3x3_b = BasicConv2d(n3x3red, n3x3,
#                                     kernel_size=3, padding=1)
#
#         # 1x1 conv -> 3x3 conv -> 3x3 conv branch
#         self.b3_1x1_a = BasicConv2d(in_planes, n5x5red,
#                                     kernel_size=1)
#         self.b3_3x3_b = BasicConv2d(n5x5red, n5x5,
#                                     kernel_size=3, padding=1)
#         self.b3_3x3_c = BasicConv2d(n5x5, n5x5,
#                                     kernel_size=3, padding=1)
#
#         # 3x3 pool -> 1x1 conv branch
#         self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
#         self.b4_1x1 = BasicConv2d(in_planes, pool_planes,
#                                   kernel_size=1)
#
#     def forward(self, x):
#         y1 = self.b1(x)
#         y2 = self.b2_3x3_b(self.b2_1x1_a(x))
#         y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
#         y4 = self.b4_1x1(self.b4_pool(x))
#         # y的维度为[batch_size, out_channels, C_out,L_out]
#         # 合并不同卷积下的特征图
#         return torch.cat([y1, y2, y3, y4], 1)
#
#
# class ResBlk(nn.Module):
#
#     def __init__(self, ch_in, ch_out, stride=1):
#         # 通过stride减少参数维度
#         super(ResBlk, self).__init__()
#
#         self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(ch_out)
#
#         # [b, ch_in, h, w] => [b, ch_out, h, w]
#         self.extra = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
#             nn.BatchNorm2d(ch_out)
#         )
#
#     def forward(self, x):
#         '''
#         :param x: [b, ch, h, w]
#         :return:
#         '''
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         # short cut
#         # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
#         # element-wise add:
#         out = self.extra(x) + out
#
#         return out
# lenet
class CNN_v1(nn.Module):
    def __init__(self, args):
        super(CNN_v1, self).__init__()
        # 定义卷积层，1个输入通道，6个输出通道，3*3的卷积filter，外层补上了两圈0,所以输入的是10x10
        self.conv1 = nn.Conv2d(args.num_channels, 6, 3, padding=2)
        # 第二个卷积层，6个输入，16个输出，3*3的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 3, padding=2)

        # 最后是三个全连接层
        self.fc1 = nn.Linear(16 * 3 * 3, 72)
        self.fc2 = nn.Linear(72, 36)
        self.fc3 = nn.Linear(36, args.num_classes)

    def forward(self, x):
        '''前向传播函数'''
        # 先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # print('size', x.size())
        # 第一个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# alexnet

class CNN_v2(nn.Module):
    def __init__(self, args):
        super(CNN_v2, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*28*28
            nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
# resnet
class CNN_v3(nn.Module):
    def __init__(self,args):
        super(CNN_v3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h/2, w/2] => [b, 256, h/4, w/4]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h/4, w/4] => [b, 512, h/8, w/8]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.out_layer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # [b, 512, h/16, w/16] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # [b, 512, 1, 1] => [b, 512]
        x = x.view(x.size(0), -1)
        # [b, 512] => [b, 10]
        x = self.out_layer(x)

        return x
# vgg
class CNN_v4(nn.Module):

    def __init__(self, args):
        super(CNN_v4, self).__init__()
        self.conv1_1 = nn.Conv2d(args.num_channels, 64, 5)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out

