import torch
import torch.nn as nn

class Side(nn.Module):
    def __init__(self,channels, is_bn=False,is_act=True):
        super(Side,self).__init__()
        self.channels = channels
        self.is_bn = is_bn
        self.is_act = is_act
