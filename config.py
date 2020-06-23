from easydict import EasyDict as edict
import torchvision.transforms as transforms

# =========================   参数设置   ==========================
opt = edict()
opt.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =========================   数据读取   ==========================
opt.read_data = edict()
opt.read_data.train = edict()
opt.read_data.train.file_path = "./Data/train.txt"
opt.read_data.train.transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
opt.read_data.train.batch_size = 128
opt.read_data.train.shuffle = True

opt.read_data.test = edict()
opt.read_data.test.file_path = "./Data/test.txt"
opt.read_data.test.transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
opt.read_data.test.batch_size = 500
opt.read_data.test.shuffle = False

# ========================   训练       ============================
opt.module_train = edict()
opt.module_train.max_epoch = 120
opt.module_train.margin_s = 1.0
opt.module_train.margin_m = 0.0
opt.lr_mul = [20,35,50,65,80,
              95,110,120,130,140,
              150,160,170,180,190]
opt.lr_gamma = 0.5

# ========================   保存       ============================
opt.module_save = edict()
opt.module_save.path = './model/'

