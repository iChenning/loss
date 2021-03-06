import os
import torch
import time
from datetime import datetime
from tensorboardX import SummaryWriter


class Log():
    def __init__(self, opt):
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
        if opt.log_name == None:
            self.log_dir = os.path.join('log',
                                        opt.train.feature_net + '-' + opt.train.fc_type + '-' + opt.train.loss_type + '_' + time_str)
        else:
            self.log_dir = os.path.join('log',
                                        opt.train.feature_net + opt.log_name + '-' + opt.train.fc_type + '-' + opt.train.loss_type)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        if opt.train.fc_type == 'Dot':
            self.scale = 1.0
        else:
            self.scale = opt.train.scale
        self.device = opt.device

        self.best_acc = 0.6

    def init(self):
        self.correct = 0
        self.total = 0
        self.time_start = time.time()

    def update(self, x, label):
        self.time_now = time.time()

        _, predicted = torch.max(x.data, 1)
        self.correct += predicted.eq(label.data).cpu().sum().item()
        self.total += label.size(0)

        one_hot = torch.zeros(x.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        x_s = x * one_hot
        self.pos_x_max = torch.max(x_s) / self.scale
        self.pos_x_min = torch.min(x_s) / self.scale
        x_s = x * (1.0 - one_hot)
        self.neg_x_max = torch.max(x_s) / self.scale
        self.neg_x_min = torch.min(x_s) / self.scale

    def log_train(self, optimizer, loss_info, opt, i_epoch, i_iter, trainloader_len):
        if len(loss_info) > 1:
            print("Training: Epoch[{:0>3}/{:0>3}] "
                  "Iteration[{:0>3}/{:0>3}] "
                  "Loss: {:.4f} "
                  "Loss_soft: {:.4f} "
                  "Loss_center: {:.4f} "
                  "Acc:{:.3%} "
                  "Time:{:.2f}h".format(
                i_epoch + 1, opt.train.max_epoch,
                i_iter + 1, trainloader_len,
                loss_info[0].item(),
                loss_info[1].item(),
                loss_info[2].item(),
                self.correct / self.total,
                (self.time_now - self.time_start) / (i_iter + 1) * trainloader_len * (opt.train.max_epoch - i_epoch) / 3600))
            self.writer.add_scalars('Loss_group', {'train_loss': loss_info[0]}, i_epoch * trainloader_len + i_iter)
            self.writer.add_scalars('Loss_group', {'train_loss_soft': loss_info[1]}, i_epoch * trainloader_len + i_iter)
            self.writer.add_scalars('Loss_group', {'train_loss_center': loss_info[2]},
                                    i_epoch * trainloader_len + i_iter)
        else:
            print("Training: Epoch[{:0>3}/{:0>3}] "
                  "Iteration[{:0>3}/{:0>3}] "
                  "lr: {:.4f}"
                  "Loss: {:.4f} "
                  "Acc:{:.3%} "
                  "Time:{:.2f}h".format(
                i_epoch + 1, opt.train.max_epoch,
                i_iter + 1, trainloader_len,
                optimizer.state_dict()['param_groups'][0]['lr'],
                loss_info[0].item(),
                self.correct / self.total,
                (self.time_now - self.time_start) / (i_iter + 1) * trainloader_len * (opt.train.max_epoch - i_epoch) / 3600))
            self.writer.add_scalars('Loss_group', {'train_loss': loss_info[0]}, i_epoch * trainloader_len + i_iter)

        self.writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'],
                               i_epoch * trainloader_len + i_iter)
        self.writer.add_scalars('Accuracy_group', {'train_acc': self.correct / self.total},
                                i_epoch * trainloader_len + i_iter)

        self.writer.add_scalars('x_group', {'pos_x_max': self.pos_x_max}, i_epoch * trainloader_len + i_iter)
        self.writer.add_scalars('x_group', {'pos_x_min': self.pos_x_min}, i_epoch * trainloader_len + i_iter)
        self.writer.add_scalars('x_group', {'neg_x_max': self.neg_x_max}, i_epoch * trainloader_len + i_iter)
        self.writer.add_scalars('x_group', {'neg_x_min': self.neg_x_min}, i_epoch * trainloader_len + i_iter)

    def log_test(self, net, opt, i_epoch, trainloader_len):
        acc = self.correct / self.total
        print("valid acc:{:.3%},  best acc:{:.3%} ".format(acc, max(acc, self.best_acc)))
        self.writer.add_scalars('Accuracy_group', {'test_acc': acc}, (i_epoch + 1) * trainloader_len)

        if acc > self.best_acc:
            f_best_acc = open(self.log_dir + "/best_acc.txt", 'a')
            f_best_acc.write("EPOCH=%d,best_acc= %.3f%%\n" % (i_epoch + 1, acc * 100.0))
            f_best_acc.close()
            self.best_acc = acc

            print('Saving model......')
            torch.save(net.state_dict(), '%s/best_net.pth' % (self.log_dir))
