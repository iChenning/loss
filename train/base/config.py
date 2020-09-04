from easydict import EasyDict as edict
import torch

# =========================   参数设置   ==========================
opt = edict()

# ========================   训练&验证   ============================
opt.train = edict()
opt.train.max_epoch = 200
opt.train.lr_init = 0.1
opt.train.lr_mul = [75, 130, 170]
opt.train.lr_gamma = 0.1

opt.is_side1 = False  # True False
opt.is_side2 = False  # True False
opt.is_side3 = False  # True False
opt.log_name = 'shiyan'  # None

opt.train.feature_net = 'Resnet50'  # 'Net5' 'Net5_Side' 'Res18_SideNew' 'Resnet22' 'Res22_Side' 'Resnet26' 'ACRes26'
fc_type = 'Dot'  # 'Dot' 'Cos' 'CosAddMargin'
if fc_type == 'Dot':
    opt.train.fc_type = 'Dot'
elif fc_type == 'Cos':
    opt.train.fc_type = 'Cos'
    opt.train.scale = 100.0
elif fc_type == 'CosAddMargin':
    opt.train.fc_type = 'CosAddMargin'
    opt.train.scale = 100.0
    opt.train.margin = 0.2
opt.train.loss_type = 'standard'  # 'standard' 'add_center'

opt.train.is_net_load = False  # True False
if opt.train.is_net_load:
    opt.train.net_path = 'log/Resnet22-Dot-standard/best_net.pth'
else:
    opt.train.net_path = None


# =========================   数据读取   ==========================
opt.data = edict()

opt.data.train = edict()
opt.data.train.file_path = "/opt/code/Data/cifar10/train.txt"
opt.data.train.batch_size = 128
opt.data.train.shuffle = True

opt.data.valid = edict()
opt.data.valid.file_path = "/opt/code/Data/cifar10/valid.txt"
opt.data.valid.batch_size = 128
opt.data.valid.shuffle = False

opt.data.test = edict()
opt.data.test.file_path = "/opt/code/Data/cifar10/test.txt"
opt.data.test.batch_size = 128
opt.data.test.shuffle = False


# ========================   GPU相关设置 ==========================
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.device_count() >= 4:
    torch.cuda.set_device(3)  # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

opt.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')