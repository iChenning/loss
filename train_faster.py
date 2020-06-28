from utils.my_dataset import MyDataset, data_prefetcher
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import argparse
from importlib.machinery import SourceFileLoader
from moduls.modul_net5 import Net5
from moduls.modul_resnet22 import ResNet22
from moduls.fc_weight import Dot, Cos, CosAddMargin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ========================    开始训练    ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loss training with Pytorch")
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), args.config
    opt = SourceFileLoader('module.name', args.config).load_module().opt

    # ========================    数据读取    =========================
    read_train = opt.read_data.train
    # trainset = MyDataset(txt_path=read_train.file_path, transform=read_train.transforms)
    trainset = torchvision.datasets.CIFAR10(root='./Data', train=True, download=False, transform=read_train.transforms)  # 训练数据集
    trainloader = DataLoader(trainset, batch_size=read_train.batch_size, shuffle=read_train.shuffle)
    read_test = opt.read_data.test
    # testset = MyDataset(txt_path=read_test.file_path, transform=read_test.transforms)
    testset = torchvision.datasets.CIFAR10(root='./Data', train=False, download=False, transform=read_test.transforms)
    testloader = DataLoader(testset, batch_size=read_test.batch_size, shuffle=read_test.shuffle)

    # ========================    导入网络    ========================
    if opt.train.net == 'Net5':
        net = Net5(opt).to(device)
    else:
        net = ResNet22().to(device)

    if opt.train.fc_type == 'Cos':
        fc = Cos()
    elif opt.train.fc_type == 'CosAddMargin':
        fc = CosAddMargin()
    else:
        fc = Dot()

    # ========================    初始化优化器 =======================
    if 'Margin' in opt.train.fc_type:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join('log', time_str + '_' + opt.train.net + '-' + opt.train.fc_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ========================   训练及测试   =======================
    best_acc = 0.60
    for i_epoch in range(opt.train.max_epoch):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        optimizer.zero_grad()
        scheduler.step()

        trainloader_pre = data_prefetcher(trainloader)
        img, label = trainloader_pre.next()
        i_iter = 0
        while img is not None:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            feature = net(img)
            x = fc(feature, label, is_train=True)

            loss = criterion(x, label)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(x.data, 1)
            correct += predicted.eq(label.data).cpu().sum().item()
            total += label.size(0)

            print("Training: Epoch[{:0>3}/{:0>3}] "
                  "Iteration[{:0>3}/{:0>3}] "
                  "Loss: {:.4f} "
                  "Acc:{:.3%} ".format(
                i_epoch + 1, opt.train.max_epoch,
                i_iter + 1, len(trainloader),
                sum_loss / (i_iter + 1),
                correct / total))

            writer.add_scalars('Loss_group', {'train_loss': sum_loss / (i_iter + 1)}, i_epoch * len(trainloader) + i_iter)
            writer.add_scalar('learning rate', scheduler.get_lr()[0], i_epoch * len(trainloader) + i_iter)
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, i_epoch * len(trainloader) + i_iter)
            one_hot = torch.zeros(x.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            x_s = x * one_hot
            writer.add_scalars('x_group', {'pos_x_max': torch.max(x_s)}, i_epoch * len(trainloader) + i_iter)
            writer.add_scalars('x_group', {'pos_x_min': torch.min(x_s)}, i_epoch * len(trainloader) + i_iter)
            x_s = x * (1.0 - one_hot)
            writer.add_scalars('x_group', {'neg_x_max': torch.max(x_s)}, i_epoch * len(trainloader) + i_iter)
            writer.add_scalars('x_group', {'neg_x_min': torch.min(x_s)}, i_epoch * len(trainloader) + i_iter)

            img, label = trainloader_pre.next()
            i_iter += 1

        print("测试...")
        with torch.no_grad():
            correct = 0
            total = 0
            testloader_pre = data_prefetcher(testloader)
            img, label = testloader_pre.next()
            while img is not None:
                net.eval()
                img, label = img.to(device), label.to(device)
                feature = net(img)
                x = fc(feature, label, is_train=False)

                _, predicted = torch.max(x.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                img, label = testloader_pre.next()

            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = correct / total

            writer.add_scalars('Accuracy_group', {'test_acc': acc}, (i_epoch + 1) * (len(trainloader)))

            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.module_save.path, i_epoch + 1))

            if acc > best_acc:
                f_best_acc = open(log_dir + "/best_acc.txt", 'w')
                f_best_acc.write("EPOCH=%d,best_acc= %.3f%%" % (i_epoch + 1, acc * 100.0))
                f_best_acc.close()
                best_acc = acc

    print("训练完成")
