import torch
from easydict import EasyDict as edict
import torchvision.transforms as transforms
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from models import Modules
import torch.nn as nn
import torch.optim as optim
from loss import Loss, AddMarginProduct
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========================   参数设置   ==========================
opt = edict()
opt.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

opt.read_data = edict()
opt.read_data.train = edict()
opt.read_data.train.file_path = "./Data/train.txt"
opt.read_data.train.transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
opt.read_data.train.batch_size = 500
opt.read_data.train.shuffle = True

opt.read_data.test = edict()
opt.read_data.test.file_path = "./Data/test.txt"
opt.read_data.test.transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
opt.read_data.test.batch_size = 500
opt.read_data.test.shuffle = False

opt.module_train = edict()
opt.module_train.max_epoch = 100


opt.module_save = edict()
opt.module_save.path = './model/'

#========================    数据读取    =========================
read_train = opt.read_data.train
trainset = MyDataset(txt_path=read_train.file_path, transform=read_train.transforms)
trainloader = DataLoader(trainset,batch_size=read_train.batch_size, shuffle=read_train.shuffle)
read_test = opt.read_data.test
testset = MyDataset(txt_path=read_test.file_path, transform=read_test.transforms)
testloader = DataLoader(testset, batch_size=read_test.batch_size, shuffle=read_test.shuffle)

#========================    导入网络    ========================
net = Modules(opt).to(device)

#========================    初始化优化器 =======================
XX = AddMarginProduct(s=10, m=0.01)
criterion = nn.CrossEntropyLoss()
# criterion = Loss(s=100, m=0.25)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

#========================    开始训练    ========================
if __name__=="__main__":
    with open("acc.txt", 'w') as f_acc:
        with open("log.txt", 'w') as f_log:
            best_acc = 0.0
            for i_epoch in range(opt.module_train.max_epoch):
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i_iter, data in enumerate(trainloader):
                    img, label = data
                    img, label = img.to(device), label.to(device)
                    optimizer.zero_grad()

                    x = net(img)
                    x = XX(x, label, i_epoch)
                    loss = criterion(x, label)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()

                    _, predicted = torch.max(x.data, 1)
                    correct += predicted.eq(label.data).cpu().sum()
                    total += label.size(0)

                    print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>4}/{:0>4}] Loss: {:.4f} Acc:{:.3%} Max_x:{:.4f}".format(
                        i_epoch + 1, opt.module_train.max_epoch, i_iter + 1, len(trainloader), sum_loss / (i_iter + 1), correct / total, torch.max(x)))
                    f_log.write('%3d %5d | Loss: %.04f | Acc: %.03f\n' %
                                (i_epoch + 1, i_iter + 1, sum_loss / (i_iter + 1), 100.*correct / total))
                    f_log.flush()

                print("测试")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        img, label = data
                        img, label = img.to(device), label.to(device)
                        x = net(img)

                        _, predicted = torch.max(x.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.module_save.path, i_epoch + 1))
                    f_acc.write("EPOCH=%03d,Accuracy= %.3f%%" % (i_epoch + 1, acc))
                    f_acc.write('\n')
                    f_acc.flush()

                    if acc > best_acc:
                        f_best_acc = open("best_acc.txt", 'w')
                        f_best_acc.write("EPOCH=%d,best_acc= %.3f%%" % (i_epoch + 1, acc))
                        f_best_acc.close()
                        best_acc = acc
            print("训练完成")