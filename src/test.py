import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



def tes_img(net_g, datatest, args):
    net_g.eval()
    
    # testing
    test_loss = 0
    correct = 0
    correct_test = 0
    correct_every_fault = [[0 for i in range(7)] for j in range(7)]
    for i in range(0,7):
        # data_loader.targets = []
        data_loader = DataLoader(datatest[i], batch_size=args.bs)
        l = len(data_loader)
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                #这里DataLoader中换不同的数据集target不会变化，因此手动更新，目前target用的是0,1,2,3,4,5,6，后续这里要注意sample和img会自动更新
                target = torch.tensor([i for j in range(len(target))])
                if args.gpu != -1:
                    data, target = data.to(args.device), target.to(args.device)

                log_probs = net_g(data)
                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                #测试相应类型正确率
                # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
                #测试各种故障类型的概率，不论对错，用于绘制混淆矩阵
                for j in range(len(y_pred)):
                    correct_every_fault[i][y_pred[j].item()] += round(float(1/len(datatest[i])),5)
                    if y_pred[j].item()== i :
                        correct_test+=1
    len_datatest = 0
    for i in range(0,7):
        len_datatest += len(datatest[i])

    test_loss /= len_datatest
    # accuracy = 100.00 * correct / len_datatest
    accuracy = 100.00*correct_test/len_datatest
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct_test, len_datatest, accuracy))
    return accuracy, test_loss,correct_every_fault

