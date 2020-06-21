import torch
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from models import Modules, Res5
import torch.nn as nn
import torch.optim as optim
from loss import AddMarginLinear
from config import opt
import torchvision.models as models
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import os


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ========================    数据读取    =========================
read_train = opt.read_data.train
trainset = MyDataset(txt_path=read_train.file_path, transform=read_train.transforms)
trainloader = DataLoader(trainset, batch_size=read_train.batch_size, shuffle=read_train.shuffle)
read_test = opt.read_data.test
testset = MyDataset(txt_path=read_test.file_path, transform=read_test.transforms)
testloader = DataLoader(testset, batch_size=read_test.batch_size, shuffle=read_test.shuffle)

# ========================    导入网络    ========================
net = Modules(opt).to(device)
# net = Res5(opt).to(device)
# net = models.resnet18(pretrained=False, num_classes=128).to(device)
fc = AddMarginLinear(s=10.0, m=0.01)

# ========================    初始化优化器 =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join('log', time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# ========================    开始训练    ========================
if __name__ == "__main__":
    best_acc = 0.0
    for i_epoch in range(opt.module_train.max_epoch):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        scheduler.step()
        for i_iter, data in enumerate(trainloader):
            time_s = time.time()

            img, label = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            feature = net(img)
            x = fc(feature, label, i_epoch, is_train=True)

            # if i_epoch % 2 == 0:
            #     loss = criterion(x, label)*0.1
            # else:
            #     loss = criterion(x, label)
            loss = criterion(x, label)
            loss.backward()
            optimizer.step()
            time_e = time.time()
            # print("time:", time_e - time_s)

            sum_loss += loss.item()
            _, predicted = torch.max(x.data, 1)
            correct += predicted.eq(label.data).cpu().sum().item()
            total += label.size(0)

            print(
                "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>4}/{:0>4}] Loss: {:.4f} Acc:{:.3%} Max_x:{:.2f} Min_x:{:.2f}".format(
                    i_epoch + 1, opt.module_train.max_epoch, i_iter + 1, len(trainloader), sum_loss / (i_iter + 1),
                    correct / total, torch.max(x), torch.min(x)))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': sum_loss / (i_iter + 1)},
                               i_epoch * (len(trainloader)) + i_iter)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], i_epoch * (len(trainloader)) + i_iter)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, i_epoch * (len(trainloader)) + i_iter)

            if i_epoch % 2 == 0:
                writer.add_scalars('x_group', {'x_max': torch.max(x) * 0.1}, i_epoch * (len(trainloader)) + i_iter)
                writer.add_scalars('x_group', {'x_min': torch.min(x) * 0.1}, i_epoch * (len(trainloader)) + i_iter)
            else:
                writer.add_scalars('x_group', {'x_max': torch.max(x)}, i_epoch * (len(trainloader)) + i_iter)
                writer.add_scalars('x_group', {'x_min': torch.min(x)}, i_epoch * (len(trainloader)) + i_iter)

        print("测试")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                img, label = data
                img, label = img.to(device), label.to(device)
                feature = net(img)
                x = fc(feature, label, i_epoch, is_train=False)

                _, predicted = torch.max(x.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = correct / total

            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'test_acc': acc}, (i_epoch + 1) * (len(trainloader)))

            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.module_save.path, i_epoch + 1))

            if acc > best_acc:
                f_best_acc = open("best_acc.txt", 'w')
                f_best_acc.write("EPOCH=%d,best_acc= %.3f%%" % (i_epoch + 1, acc))
                f_best_acc.close()
                best_acc = acc
    print("训练完成")
