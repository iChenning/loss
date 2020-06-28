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
opt.train.net = 'Resnet22' # 'Net5'
opt.train.fc_type = 'Dot' # 'Cos' 'CosAddMargin'
opt.train.margin_s = 30.0
opt.train.margin_m = 0.01
opt.inter = 1

opt.train.max_epoch = 150
opt.lr_mul = [20, 35, 50, 65, 80,
              95, 110, 120, 130, 140,
              150, 160, 170, 180, 190]
opt.lr_gamma = 0.5

opt.is_softmax = True


# ========================   保存       ============================
opt.module_save = edict()
opt.module_save.path = './model/'

