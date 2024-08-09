import os

from model import vgg, resnet_cs
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='train-teacher-network')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST', 'cifar10', 'cifar100'])
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--arch', default='resnet', type=str, help='architecture to use')
parser.add_argument('--output_dir', type=str, default='./models/')
parser.add_argument('--tname', type=str, default='**')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0


if args.dataset == 'cifar10':
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

    data_train = CIFAR10(args.data,
                          transform=transform_train)
    data_test = CIFAR10(args.data,
                        train=False,
                        transform=transform_test)

    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=args.num_workers)
    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.arch == 'resnet':
        net = resnet_cs.ResNet34().to(device)
    elif args.arch == 'vgg':
        net = vgg.vgg(dataset=args.dataset, depth=16).to(device)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



#
def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    # print(len(data_train))
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()


def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            labels = labels.long()
            output = net(images)
            avg_loss += criterion(output, labels.squeeze()).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.squeeze().data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
        torch.save(net, args.output_dir + args.tnamebest)
    print('Test Avg. Loss: %f, Accuracy: %f, Best_Accuracy: %f' % (avg_loss.data.item(), acc, acc_best))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    if args.dataset == 'MNIST':
        epoch = 20
    else:
        epoch = 200
    for e in range(epoch):
        train_and_test(e)
    torch.save(net, args.output_dir + args.tname)


if __name__ == '__main__':
    main()