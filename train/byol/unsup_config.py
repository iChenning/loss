from easydict import EasyDict as edict
import torch

# =========================   参数设置   ==========================
opt = edict()

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.device_count() >= 4:
    torch.cuda.set_device(2)  # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

opt.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =========================   数据读取   ==========================
opt.read_data = edict()
opt.read_data.is_disk = True  # True False

opt.read_data.train = edict()
opt.read_data.train.file_path = "Data/train.txt"
opt.read_data.train.batch_size = 250
opt.read_data.train.shuffle = True

opt.read_data.test = edict()
opt.read_data.test.file_path = "Data/test.txt"
opt.read_data.test.batch_size = 250
opt.read_data.test.shuffle = False

# ========================   训练&测试   ============================
opt.is_side1 = False  # True False
opt.is_side2 = False  # True False
opt.is_side3 = False  # True False
opt.log_name = 'unsup_byol'  # None

opt.train = edict()
opt.train.feature_net = 'Res22_Side'  # 'Net5' 'Net5_Side' 'Resnet22' 'Res22_Side' 'Resnet26' 'ACRes26'
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
    opt.train.net_path = 'log/Resnet22-Cos-standard_07-08_19-03-38/best_net.pth'
else:
    opt.train.net_path = None

opt.train.max_epoch = 200
opt.lr_init = 0.1
opt.lr_mul = [80, 135, 170]
opt.lr_gamma = 0.1
