'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse



from models import *
from models.regnet import RegNetX_200MF
from models.resnet import ResNet18

from utils import progress_bar

#
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'



#获取参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training') #创建对象，包含将命令行解析成python数据类型所需的全部信息
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')#添加参数
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data 获取数据集，进行预处理
print('==> Preparing data..')
#transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])

#transform_test = transforms.Compose([
    #transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

#trainset = torchvision.datasets.CIFAR10(
    #root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
    #trainset, batch_size=32, shuffle=True, num_workers=0)

#testset = torchvision.datasets.CIFAR10(
    #root='./data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(
    #testset, batch_size=25, shuffle=False, num_workers=0)

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
image_path = os.path.join(data_root, "data", "face_data")  # data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
trainset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
train_num = len(trainset)

face_list = trainset.class_to_idx
cla_dict = dict((val, key) for key, val in face_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

batch_size = 4
#nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#print('Using {} dataloader workers every process'.format(nw))

trainloader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

testset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
test_num = len(testset)
testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)


classes = ('patient', 'normal')



#classes = ('patient', 'normal')

# Model
print('==> Building model..')
# net = VGG('VGG19')
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
net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#训练模型或新建模型


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# 定义度量和优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        # #batch数据
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #   将数据移到GPU上
            inputs, targets = inputs.to(device), targets.to(device)
            # 先将optimizer梯度先置为0
            optimizer.zero_grad()
            # 模型输出
            outputs = net(inputs)
            # 计算loss
            loss = criterion(outputs, targets)
            loss.requires_grad_(True)
            # 反向传播，计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            train_loss += loss.item()
            #数据统计
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


    # Save checkpoint.
    #保存模型
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    #print(' Loss: %.3f  Acc: %.3f' % (loss, acc))
    #print("Loss:{}, Acc:{}".format(loss, acc))




#运行模型
for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)






