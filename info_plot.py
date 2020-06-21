import os
import matplotlib.pyplot as plt

log_path = 'my-loss/log.txt'
acc_path = 'my-loss/acc.txt'

log = open(log_path, 'r')
log_info = []
for line in log.readlines():
    temp = line.split('|')
    print(temp)

acc = open(acc_path, 'r')
