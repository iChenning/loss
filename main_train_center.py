import torch
from utils.my_dataset import MyDataset
from torch.utils.data import DataLoader
from models_resnet50 import resnet50
import torch.nn as nn
import torch.optim as optim
from moduls.fc_weight import CenterLoss
# from config import opt
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import argparse
from importlib.machinery import SourceFileLoader

# device = torch.device("cpu")
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
    trainset = MyDataset(txt_path=read_train.file_path, transform=read_train.transforms)
    trainloader = DataLoader(trainset, batch_size=read_train.batch_size, shuffle=read_train.shuffle)
    read_test = opt.read_data.test
    testset = MyDataset(txt_path=read_test.file_path, transform=read_test.transforms)
    testloader = DataLoader(testset, batch_size=read_test.batch_size, shuffle=read_test.shuffle)

    # ========================    导入网络    ========================
    # net = Modules(opt).to(device)
    net = resnet50().to(device)
    fc = CenterLoss()

    # ========================    初始化优化器 =======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    name = 'resnet50-softmax-center'
    log_dir = os.path.join('log', name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ========================   训练及测试   =======================
    best_acc = 0.8
    for i_epoch in range(opt.module_train.max_epoch):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        scheduler.step()
        for i_iter, data in enumerate(trainloader):
            img, label = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            feature = net(img)
            x, loss_center = fc(feature, label, is_train=True)
            loss_softmax = criterion(x, label)
            loss = loss_softmax + loss_center
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(x.data, 1)
            correct += predicted.eq(label.data).cpu().sum().item()
            total += label.size(0)

            print("Training: Epoch[{:0>3}/{:0>3}] "
                  "Iteration[{:0>4}/{:0>4}] "
                  "Loss: {:.4f} "
                  "Acc:{:.3%} "
                  "Max_x:{:.2f} "
                  "Min_x:{:.2f}".format(
                i_epoch + 1, opt.module_train.max_epoch,
                i_iter + 1, len(trainloader),
                sum_loss / (i_iter + 1),
                correct / total,
                torch.max(x),
                torch.min(x)))

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

        print("测试...")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                img, label = data
                img, label = img.to(device), label.to(device)
                feature = net(img)
                x = fc(feature, label, is_train=False)

                _, predicted = torch.max(x.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = correct / total

            writer.add_scalars('Accuracy_group', {'test_acc': acc}, (i_epoch + 1) * (len(trainloader)))

            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.module_save.path, i_epoch + 1))

            if acc > best_acc:
                f_best_acc = open("best_acc.txt", 'w')
                f_best_acc.write("EPOCH=%d,best_acc= %.3f%%" % (i_epoch + 1, acc * 100.0))
                f_best_acc.close()
                best_acc = acc
    print("训练完成")
