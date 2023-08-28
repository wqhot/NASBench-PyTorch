import torch
from torch import nn
import sys
sys.path.append('..')
from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec
from nasbench_pytorch.trainer import train, test

class Args():
    def __init__(self):
        self.random_state = 1
        self.data_root = './data/'
        self.in_channels = 3
        self.stem_out_channels = 128
        self.num_stacks = 3
        self.num_modules_per_stack = 3
        self.batch_size = 256
        self.test_batch_size = 256
        self.epochs = 108
        self.validation_size = 10000
        self.num_workers = 0
        self.learning_rate = 0.2
        self.optimizer = 'rmsprop'
        self.rmsprop_eps = 1.0
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.grad_clip = 5
        self.grad_clip_off = False
        self.batch_norm_momentum = 0.997
        self.batch_norm_eps = 1e-5
        self.load_checkpoint = ''
        self.num_labels = 10
        self.device = 'cpu'
        self.print_freq = 100
        self.tf_like = False

def load_network():
    args = Args()

    matrix = [[0, 1, 1, 0, 0], # 输入层
          [0, 0, 1, 0, 1], # 1x1卷积
          [0, 0, 0, 1, 0], # 3x3卷积
          [0, 0, 0, 0, 1], # 1x1卷积
          [0, 0, 0, 0, 0]]  # output layer

    operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output']

    spec = ModelSpec(matrix, operations)

    net = Network(spec, num_labels=args.num_labels, in_channels=args.in_channels,
                  stem_out_channels=args.stem_out_channels, num_stacks=args.num_stacks,
                  num_modules_per_stack=args.num_modules_per_stack,
                  momentum=args.batch_norm_momentum, eps=args.batch_norm_eps, tf_like=args.tf_like)
    return net

def load_data():
    args = Args()
    # cifar10 dataset
    dataset = prepare_dataset(args.batch_size, test_batch_size=args.test_batch_size, root=args.data_root,
                              validation_size=args.validation_size, random_state=args.random_state,
                              set_global_seed=True, num_workers=args.num_workers)

    train_loader, test_loader, test_size = dataset['train'], dataset['test'], dataset['test_size']
    valid_loader = dataset['validation'] if args.validation_size > 0 else None
    return train_loader, test_loader, valid_loader, test_size

def load_checkpoint(net, file):
    ckpt = torch.load(file, map_location=torch.device('cpu'))
    net.load_state_dict(ckpt)
    net.to(torch.device('cpu'))
    return net, ckpt

def save_network(net, file):
    torch.save(net, file)

def test_network(net, test_loader, test_size):
    criterion = nn.CrossEntropyLoss()
    # test(net, test_loader, device=torch.device('cpu'))
    result = test(net, test_loader, loss=criterion, num_tests=test_size, device=torch.device('cpu'))
    return result
