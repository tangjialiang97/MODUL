import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def mixup_constrastive_loss(input, input_aug, target, model_s, model_t, embed_s, embed_t, prototypes):
    alpha = 8
    temperature = 0.3
    batch_size = input.size(0)
    num_classes = 10
    L = np.random.beta(alpha, alpha)
    labels = torch.zeros(batch_size, num_classes).cuda().scatter_(1, target.view(-1, 1), 1)

    inputs = torch.cat([input, input_aug], dim=0)
    idx = torch.randperm(batch_size * 2)
    labels = torch.cat([labels, labels], dim=0)

    input_mix = L * inputs + (1 - L) * inputs[idx]
    labels_mix = L * labels + (1 - L) * labels[idx]

    _, feat_mix_t, _, _ = model_t(input_mix, out_feature=True)
    _, feat_mix_s, _, _ = model_s(input_mix, out_feature=True)

    feat_mix_t = embed_t(feat_mix_t)
    feat_mix_s = embed_s(feat_mix_s)

    logits_proto_t = torch.mm(feat_mix_t, prototypes.t()) / temperature
    logits_proto_s = torch.mm(feat_mix_s, prototypes.t()) / temperature
    loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels_mix, dim=1))
    loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels_mix, dim=1))

    return loss_proto_t + loss_proto_s


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def instance_constrastive_loss(feat, feat_aug, embed):
    criterions = nn.CrossEntropyLoss()
    batch_size = feat.size(0)
    temperature = 0.3
    feat = embed(feat)
    feat_aug = embed(feat_aug)
    sim_clean = torch.mm(feat, feat.t())
    mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
    sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

    sim_aug = torch.mm(feat, feat_aug.t())
    sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

    logits_pos = torch.bmm(feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)).squeeze(-1)
    logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(batch_size).long().cuda()

    loss_instance = criterions(logits/temperature, instance_labels)
    return loss_instance

# def instance_constrastive_loss(feat, feat_aug):
#     criterions = nn.CrossEntropyLoss()
#     batch_size = feat.size(0)
#     temperature = 0.3
#     sim_clean = torch.mm(feat, feat.t())
#     mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
#     sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)
#
#     sim_aug = torch.mm(feat, feat_aug.t())
#     sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)
#
#     logits_pos = torch.bmm(feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)).squeeze(-1)
#     logits_neg = torch.cat([sim_clean, sim_aug], dim=1)
#
#     logits = torch.cat([logits_pos, logits_neg], dim=1)
#     instance_labels = torch.zeros(batch_size).long().cuda()
#
#     loss_instance = criterions(logits/temperature, instance_labels)
#     return loss_instance
#

# def mixup_constrastive_loss(input, input_aug, target, num_classes, model, prototypes):
#     alpha = 8
#     temperature = 0.3
#     batch_size = input.size(0)
#     L = np.random.beta(alpha, alpha)
#     labels = torch.zeros(batch_size, num_classes).cuda().scatter_(1, target.view(-1, 1), 1)
#     inputs = torch.cat([input, input_aug], dim=0)
#     idx = torch.randperm(batch_size*2)
#     labels = torch.cat([labels, labels], dim=0)
#
#     input_mix = L * inputs + (1 - L) * inputs[idx]
#     labels_mix = L * labels + (1 - L) * labels[idx]
#
#     _, feat_mix, _, _ = model(input_mix, out_feature=True)
#     logits_proto = torch.mm(feat_mix, prototypes.t()) / temperature
#     loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
#     return loss_proto



