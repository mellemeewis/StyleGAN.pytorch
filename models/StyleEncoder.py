from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

import torch
from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad, RandomHorizontalFlip
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU, MSELoss
from torch.optim import Adam

from models.Blocks import EncoderBlock


class StyleEncoder(nn.Module):

    def __init__(self, in_size, channels, zchannels, zs=256, k=3, unmapping=8, batch_norm=False, z_dropout=0.25):
        super().__init__()

        c, h, w = in_size
        c1, c2, c3, c4, c5 = channels
        z0, z1, z2, z3, z4, z5 = zchannels

        # resnet blocks
        self.block1 = EncoderBlock(c,  c1, kernel_size=k, batch_norm=batch_norm)
        self.block2 = EncoderBlock(c1, c2, kernel_size=k, batch_norm=batch_norm)
        self.block3 = EncoderBlock(c2, c3, kernel_size=k, batch_norm=batch_norm)
        self.block4 = EncoderBlock(c3, c4, kernel_size=k, batch_norm=batch_norm)
        self.block5 = EncoderBlock(c4, c5, kernel_size=k, batch_norm=batch_norm)

        self.affine0 = nn.Linear(util.prod(in_size), 2 * zs)
        self.affine1 = nn.Linear(util.prod((c1, h//2, w//2)), 2 * zs)
        self.affine2 = nn.Linear(util.prod((c2, h//4, w//4)), 2 * zs)
        self.affine3 = nn.Linear(util.prod((c3, h//8, w//8)), 2 * zs)
        self.affine4 = nn.Linear(util.prod((c4, h//16, w//16)), 2 * zs)
        self.affine5 = nn.Linear(util.prod((c5, h//32, w//32)), 2 * zs)

        self.affinez = nn.Linear(12 * zs, 2 * zs)

        # 1x1 convolution to distribution on "noise space"
        # (mean and sigma)
        self.tonoise0 = nn.Conv2d(c,  z0*2, kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(c1, z1*2, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(c2, z2*2, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(c3, z3*2, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(c4, z4*2, kernel_size=1, padding=0)
        self.tonoise5 = nn.Conv2d(c5, z5*2, kernel_size=1, padding=0)

        self.z_dropout = nn.Dropout2d(p=z_dropout, inplace=False)



        um = []
        for _ in range(unmapping):
            um.append(nn.ReLU())
            um.append(nn.Linear(zs*2, zs*2))
        self.unmapping = nn.Sequential(*um)

    def forward(self, x0, depth):
        b = x0.size(0)

        n0 = n1 = n2 = n3 = n4 = n5 = None

        z0 = self.affine0(x0.view(b, -1))
        x0 = util.F.instance_norm(x0)
        n0 = self.tonoise0(x0)

        if depth <= 0:
            z=z0
            z = self.unmapping(z)
            return z, n0, n1, n2, n3, n4, n5

        x1 = F.avg_pool2d(self.block1(x0), 2)
        z1 = self.affine1(x1.view(b, -1))
        x1 = util.F.instance_norm(x1)
        n1 = self.tonoise1(x1)

        if depth <= 1:
            zbatch = torch.cat([z0[:, None, :],z1[:, None, :]], dim=1)
            z = self.z_dropout(zbatch)       
            if z[z != 0].sum() ==0:
                z = zbatch
            z = z.sum(dim=1)
            z = self.unmapping(z)
            return z, n0, n1, n2, n3, n4, n5

        x2 = F.avg_pool2d(self.block2(x1), 2)
        z2 = self.affine2(x2.view(b, -1))
        x2 = util.F.instance_norm(x2)
        n2 = self.tonoise2(x2)

        if depth <= 2:
            zbatch = torch.cat([z0[:, None, :],z1[:, None, :],z2[:, None, :]], dim=1)
            z = self.z_dropout(zbatch)
            if z[z != 0].sum() == 0:
                z = zbatch       
            z = z.sum(dim=1)
            z = self.unmapping(z)
            return z, n0, n1, n2, n3, n4, n5

        x3 = F.avg_pool2d(self.block3(x2), 2)
        z3 = self.affine3(x3.view(b, -1))
        x3 = util.F.instance_norm(x3)
        n3 = self.tonoise3(x3)

        if depth <= 3:
            zbatch = torch.cat([z0[:, None, :],z1[:, None, :],z2[:, None, :], z3[:, None, :]], dim=1)
            z = self.z_dropout(zbatch)   
            if z[z != 0].sum() == 0:
                z = zbatch   
            z = z.sum(dim=1)
            z = self.unmapping(z)
            return z, n0, n1, n2, n3, n4, n5

        x4 = F.avg_pool2d(self.block4(x3), 2)
        z4 = self.affine4(x4.view(b, -1))
        x4 = util.F.instance_norm(x4)
        n4 = self.tonoise4(x4)

        if depth <= 4:
            zbatch = torch.cat([z0[:, None, :],z1[:, None, :],z2[:, None, :], z3[:, None, :], z4[:, None, :]], dim=1)
            z = self.z_dropout(zbatch)     
            if z[z != 0].sum() == 0:
                z = zbatch
            z = z.sum(dim=1)
            z = self.unmapping(z)
            return z, n0, n1, n2, n3, n4, n5

        x5 = F.avg_pool2d(self.block5(x4), 2)
        z5 = self.affine5(x5.view(b, -1))
        x5 = util.F.instance_norm(x5)
        n5 = self.tonoise5(x5)

        # combine the z vectors

        zbatch = torch.cat([
            z0[:, None, :],
            z1[:, None, :],
            z2[:, None, :],
            z3[:, None, :],
            z4[:, None, :],
            z5[:, None, :]], dim=1)

        z = self.z_dropout(zbatch)    
        if z[z != 0].sum() == 0:
            z = zbatch   
        z = z.sum(dim=1)
        z = self.unmapping(z)
        return z, n0, n1, n2, n3, n4, n5
