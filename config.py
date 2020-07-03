from easydict import EasyDict as edict

import torch
from PIL import ImageFilter
import random
import numpy as np
from PIL import Image
from torchvision import transforms


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, img):
        img = Image.fromarray(img)
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(img)


# =========================   参数设置   ==========================
opt = edict()

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.device_count() >= 2:
    torch.cuda.set_device(3)  # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

opt.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =========================   数据读取   ==========================
opt.read_data = edict()
opt.read_data.is_disk = True # True False

opt.read_data.train = edict()
opt.read_data.train.file_path = "./Data/train.txt"
opt.read_data.train.transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# opt.read_data.train.transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomResizedCrop(32, scale=(0.5, 1.)),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
#     transforms.RandomGrayscale(p=.5),
#     # GaussianBlur([.1, 2.]),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

opt.read_data.train.batch_size = 250
opt.read_data.train.shuffle = True

opt.read_data.test = edict()
opt.read_data.test.file_path = "./Data/test.txt"
opt.read_data.test.transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
opt.read_data.test.batch_size = 250
opt.read_data.test.shuffle = False

# ========================   训练       ============================
opt.train = edict()
opt.train.feature_net = 'Net5'  # 'Net5' 'Resnet22' 'Resnet26' 'ACRes26'

fc_type = 'Cos' # 'Dot' 'Cos' 'CosAddMargin'
if fc_type == 'Dot':
    opt.train.fc_type = 'Dot'
elif fc_type == 'Cos':
    opt.train.fc_type = 'Cos'
    opt.train.scale = 100.0
elif fc_type == 'CosAddMargin':
    opt.train.fc_type = 'CosAddMargin'
    opt.train.scale = 100.0
    opt.train.margin = 0.2

opt.train.loss_type = 'add_center' # 'standard' 'add_center'

is_byol = False # False True
if is_byol:
    opt.train.max_epoch = 100
    opt.lr_mul = [40, 66, 85]
else:
    opt.train.max_epoch = 200
    opt.lr_mul = [80, 135, 170]
opt.lr_gamma = 0.1

# ========================   保存       ============================
opt.module_save = edict()
opt.module_save.path = './model/'
