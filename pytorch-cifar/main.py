'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import math
import os
import argparse

from models import *
from utils import progress_bar
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--warm_up', default=10, type=int, help='warm up')
parser.add_argument('--epochs', default=80, type=int, help='epochs')
parser.add_argument('--batch_size', default=128, type=int, help='epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dir', type=str, help='root dir')
parser.add_argument('--method', type=str, help='batch normal')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar_data/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.method=='official':
    net = VGG('VGG11')
elif args.method=='my_batch':
    net = VGG_My_Batch('VGG19')
elif args.method=='sm_batch':
    net = VGG_SM_Batch('VGG19')
elif args.method=='sv_batch':
    net = VGG_SV_Batch('VGG19')
elif args.method=='smv_batch':
    net = VGG_SMV_Batch('VGG19')
elif args.method=='svv_batch':
    net = VGG_SVV_Batch('VGG19')
elif args.method=='detach_var':
    net = VGG_DetachVar('VGG19')
elif args.method=='detach_l2':
    net = VGG_DetachL2('VGG19')
elif args.method == 'detach_mean':
    net = VGG_Detach_mean('VGG19')
else:
    raise NotImplementedError
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.dir+'/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)


writer = SummaryWriter(args.dir)
class AutoStep():
    def __init__(self, func, name):
        self.func = func
        self.step = 0
        self.name = name
    def write(self, val):
        self.func(self.name, val, self.step)
        self.step += 1
        
train_loss_w = AutoStep(writer.add_scalar, 'train/loss')
test_loss_w = AutoStep(writer.add_scalar, 'test/loss')
test_acc_w = AutoStep(writer.add_scalar, 'test/acc')
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_w.write(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_loss_w.write(test_loss/(batch_idx+1))
    test_acc_w.write(correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, args.dir+'/ckpt.pth')
        best_acc = acc
def adjust_learning_rate(optimizer, epoch):

    lr_min = 0.0001
    lr_max = 0.1 * args.batch_size / 256

    if epoch <= args.warm_up:
        lr = lr_min + 0.5*(lr_max - lr_min)*(1 - math.cos(epoch/(args.warm_up+1)*math.pi))
    else :
        lr = lr_min + 0.5*(lr_max - lr_min)*\
             (1 + math.cos((epoch - args.warm_up)/(args.epochs - args.warm_up)*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
for epoch in range(start_epoch, start_epoch+50):
    lr = adjust_learning_rate(optimizer,epoch)
    print("lr:",lr)
    train(epoch)
    test(epoch)
#     scheduler.step()

writer.close()