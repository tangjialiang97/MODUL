import os


from edl_losses import *
import dataloader_c32 as dataloader
from model import resnet, vgg
from loss_set import mixup_constrastive_loss, Embed, instance_constrastive_loss
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST', 'cifar10', 'cifar100'])
parser.add_argument('--noisy_dataset', type=str, default='imagenet32', choices=['MNIST', 'cifar10', 'cifar100', 'imagenet32', 'i32'])
parser.add_argument('--noise_dir', default='./noise', type=str)
parser.add_argument('--data_path', default='./data/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--noise_data_dir', default='./data/imagenet32', type=str, help='path to resized imagenet data')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--ach_t', type=str, default='resnet', choices=['vgg', 'resnet'])
parser.add_argument('--ach_s', type=str, default='resnet', choices=['vgg', 'resnet'])
parser.add_argument('--in_dim', type=int, default=512)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--depth_t', type=int, default=34)
parser.add_argument('--depth_s', type=int, default=18)
parser.add_argument('--width_t', type=int, default=2)
parser.add_argument('--width_s', type=int, default=2)
parser.add_argument('--model_width', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--on', default=0.5, type=float, help='open noise ratio')
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--numworker', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--beta', type=float, default=0.8)
parser.add_argument('--th', type=float, default=3.3)
parser.add_argument('--cross', type=float, default=0.1)
parser.add_argument('--cross_u', type=float, default=0.1)
parser.add_argument('--kd', type=float, default=0.9)
parser.add_argument('--kd_u', type=float, default=0.9)
parser.add_argument('--cs', type=float, default=0.1)
parser.add_argument('--csi', type=float, default=0.01)
parser.add_argument('--open', type=float, default=0.01)
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--output_dir', type=str, default='./models/')
parser.add_argument('--model', default='./models/***', type=str, metavar='PATH',
                    help='path to the teacher model')
parser.add_argument('--model_name', type=str, default='temp_name')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0

if args.ach_t == 'resnet':
    model_t = resnet.ResNet34().cuda()
if args.ach_t == 'vgg':
    model_t = vgg.vgg(dataset=args.dataset, depth=args.depth_t).cuda()

model_t = torch.load(args.model)
model_t.eval()
rotnet_head = torch.nn.Linear(512, 4)
rotnet_head = rotnet_head.cuda()

embed_s = Embed(args.in_dim, args.out_dim)
embed_t = Embed(args.in_dim, args.out_dim)
embed_s = embed_s.cuda()
embed_t = embed_t.cuda()

trainable_list = nn.ModuleList([])
trainable_list.append(rotnet_head)
trainable_list.append(embed_s)
trainable_list.append(embed_t)


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

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.numworker)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=args.numworker)

    if args.ach_s == 'resnet':
        model_s = resnet.ResNet18().cuda()
    if args.ach_s == 'vgg':
        model_s = vgg.vgg(depth=args.depth_s).cuda()

    trainable_list.append(model_s)
    criterion = torch.nn.CrossEntropyLoss().cuda()

transform = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

loader = dataloader.cifar_dataloader(args.dataset, args.noisy_dataset, r=args.r, on=args.on, noise_mode=args.noise_mode,
                                     batch_size=args.batch_size, num_workers=4, \
                                     root_dir=args.data_path, noise_data_dir=args.noise_data_dir,
                                     noise_file='%s/%.1f_%0.2f_%s.json' % (
                                     args.noise_dir, args.r, args.on, args.noisy_dataset))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch == 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    elif epoch == 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    elif epoch == 160:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


class Hint(nn.Module):
    '''
    FitNets: Hints for Thin Deep Nets
    https://arxiv.org/pdf/1412.6550.pdf
    '''

    def __init__(self):
        super(Hint, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(fm_s, fm_t)

        return loss


criterion_kd = Hint()
optimizer = torch.optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def get_losses_on_all(model, model_name):
    eval_loader = loader.run('eval_train')
    model.eval()
    subjective_loss = edl_mse_loss
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            if model_name == 'netS':
                y = one_hot_embedding(targets)
                y = y.cuda()
                loss = subjective_loss(outputs, y.float())
            else:
                loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    return losses


def fit_gmm_multiple_components(input_loss):
    gmm = GaussianMixture(n_components=20, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    components_open = []
    components_clean = []
    components_closed = []
    for n in range(gmm.n_components):
        if (gmm.means_[n] > .3) & (gmm.means_[n] < .9):
            components_open.extend([n])
        elif (gmm.means_[n] < .3):
            components_clean.extend([n])
        else:
            components_closed.extend([n])
    prob = gmm.predict_proba(input_loss)
    prob_clean = np.sum(prob[:, components_clean], axis=1)
    prob_closed = np.sum(prob[:, components_closed], axis=1)
    prob_open = np.sum(prob[:, components_open], axis=1)
    return prob_clean, prob_open, prob_closed

def eval_train(model, all_loss, model_name):
    losses = get_losses_on_all(model, model_name)
    all_loss.append(losses)

    if args.r == 0.9:
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    strongClean = input_loss.argmin()
    strongClosed = input_loss.argmax()

    probClean, probOpen, probClosed = fit_gmm_multiple_components(input_loss)

    predClean = (probClean > probOpen) & (probClean > probClosed)
    predClosed = (probClosed > probClean) & (probClosed > probOpen)
    predOpen = (probOpen > probClean) & (probOpen > probClosed)
    predClean[strongClean] = True
    predClosed[strongClosed] = True
    return [probClean, probOpen, probClosed], [predClean, predOpen, predClosed], all_loss


def cal_prot(model, tranloader, num_class):
    prototypes = []
    model.eval()
    for i, (images, images_aug, labels, w_x) in enumerate(tranloader):
        images, images_aug, labels = Variable(images).cuda(), Variable(images_aug).cuda(), Variable(labels).cuda()
        with torch.no_grad():
            _, feature, _, _ = model(images, out_feature=True)
            feature = embed_t(feature)
            feature = feature.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            if i == 0:
                features = np.zeros((len(tranloader.dataset), feature.shape[1]), dtype='float32')
                targets = np.zeros((len(tranloader.dataset)), dtype='int')
            if i < len(tranloader) - 1:
                features[i * args.batch_size: (i + 1) * args.batch_size] = feature
                targets[i * args.batch_size: (i + 1) * args.batch_size] = labels
            else:
                features[i * args.batch_size:] = feature
                targets[i * args.batch_size:] = labels
    for c in range(num_class):
        prototype = features[np.where(targets == c)].mean(0)  # compute prototypes with pseudo-label
        prototypes.append(torch.Tensor(prototype))
    prototypes = torch.stack(prototypes).cuda()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(
        1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def train(nepoch):
    all_loss = [[], []]
    probs, preds, all_loss[0] = eval_train(model_t, all_loss[0], 'netS')
    probClean, probOpen, probClosed = probs[0], probs[1], probs[2]
    predClean, predOpen, predClosed = preds[0], preds[1], preds[2]

    labeled_trainloader, unlabeled_trainloader, open_trainloader, labeled_dataset, unlabeled_dataset, open_dataset = loader.run(
        'trainD', predClean, predClosed,
        probClean, predOpen)
    prot = cal_prot(model_t, labeled_trainloader, 10)

    for epoch in range(nepoch):

        if args.dataset != 'MNIST':
            adjust_learning_rate(optimizer, epoch)
        global cur_batch_win

        model_s.train()
        rotnet_head.train()
        embed_t.train()
        embed_s.train()

        unlabeled_train_iter = iter(unlabeled_trainloader)
        open_train_iter = iter(open_trainloader)

        for i, (images, images_aug, labels, w_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u_aug = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u_aug = unlabeled_train_iter.next()

            try:
                inputs_o = open_train_iter.next()
            except:
                open_train_iter = iter(open_trainloader)
                inputs_o = open_train_iter.next()

            images, images_aug, labels = Variable(images).cuda(), Variable(images_aug).cuda(), Variable(labels).cuda()
            inputs_u, inputs_u_aug = Variable(inputs_u).cuda(), Variable(inputs_u_aug).cuda()
            inputs_o = Variable(inputs_o).cuda()

            optimizer.zero_grad()

            output_t, feat_t, feature_t, feature_temp_t = model_t(images, out_feature=True)
            output_s, feat_s, feature_s, feature_temp_s = model_s(images, out_feature=True)

            output_t_u, _, feature_t_u, feature_temp_t_u = model_t(inputs_u, out_feature=True)
            output_s_u, _, feature_s_u, feature_temp_s_u = model_s(inputs_u, out_feature=True)

            _, predicted = torch.max(output_t_u.data, 1)
            predicted_t = predicted.data.clone()

            inputs_r = torch.cat([torch.rot90(inputs_o, i, [2, 3]) for i in range(4)], dim=0)
            targets_r = torch.cat([torch.empty(inputs_o.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()

            _, feats_r, _, _ = model_s(inputs_r, out_feature=True)

            loss_cs_o = F.cross_entropy(rotnet_head(feats_r), targets_r, reduction='mean')

            loss_mix_s_u = mixup_constrastive_loss(inputs_u, inputs_u_aug, predicted_t, model_s, model_t, embed_s, embed_t, prot)
            loss_mix_t_u = mixup_constrastive_loss(inputs_u, inputs_u_aug, predicted_t, model_s, model_t, embed_s, embed_t, prot)
            loss_mix_s = mixup_constrastive_loss(images, images_aug, labels, model_s, model_t, embed_s, embed_t, prot)
            loss_mix_t = mixup_constrastive_loss(images, images_aug, labels, model_s, model_t, embed_s, embed_t, prot)
            loss_cs_i = instance_constrastive_loss(feat_s, feat_t, embed_s, embed_t)

            loss_cs = loss_mix_s + loss_mix_t + loss_mix_s_u + loss_mix_t_u

            loss_cross_u = F.cross_entropy(output_s_u, predicted_t)

            loss_cross = F.cross_entropy(output_s, labels)

            loss_kd = criterion_kd(feature_temp_s, feature_temp_t.detach())
            loss_kd_u = criterion_kd(feature_temp_s_u, feature_temp_t_u.detach())

            loss = loss_cross * args.cross + loss_kd * args.kd + loss_kd_u * args.kd_u + \
                   loss_cross_u * args.cross_u + loss_cs * args.cs + loss_cs_o * args.open + loss_cs_i * args.csi

            if i == 1:
                if i == 1:
                    print(
                        'Train - Epoch %d, Batch: %d, Loss_cross: %f, Loss_cross_u: %f, Loss_kd: %f, Loss_kd_u: %f, Loss_cs_o: %f, Loss_cs: %f, Loss_cs_i: %f, Loss: %f' %
                        (epoch, i, loss_cross.data.item(), loss_cross_u.item(), loss_kd.data.item(),
                         loss_kd_u.data.item(), loss_cs_o.data.item(), loss_cs.data.item(), loss_cs_i.data.item(), loss.data.item()))

            loss.backward()
            optimizer.step()

        acc_best, acc = test()
    return acc_best, acc


def test():
    global acc, acc_best
    model_s.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = model_s(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader.dataset)
    acc = float(total_correct) / len(data_test_loader.dataset)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f, Best_Accuracy: %f' % (avg_loss.data.item(), acc, acc_best))
    return acc_best, acc


def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 250

    acc_best, acc = train(epoch)


if __name__ == '__main__':
    main()