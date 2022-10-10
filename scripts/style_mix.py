import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment



ndf = 64
ngf = 64
nz = 256
nlr = 0.0002
nbeta1 = 0.5
use_cuda = True
multi_gpu = False
dataloader_workers = 8
current_iteration = 0
save_interval = 100
device = 'cuda:0'
im_size = 256


netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
netG.apply(weights_init)

netD = Discriminator(ndf=ndf, im_size=im_size)
netD.apply(weights_init)

netG.to(device)
netD.to(device)

avg_param_G = copy_G_params(netG)

fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

j = 4
checkpoint = "./models/all_%d.pth"%(j*10000)
ckpt = torch.load(checkpoint)
netG.load_state_dict(ckpt['g'])
netD.load_state_dict(ckpt['d'])
avg_param_G = ckpt['g_ema']
load_params(netG, avg_param_G)

bs = 8
noise_a = torch.randn(bs, nz).to(device)
noise_b = torch.randn(bs, nz).to(device)

def get_early_features(net, noise):
    feat_4 = net.init(noise)
    feat_8 = net.feat_8(feat_4)
    feat_16 = net.feat_16(feat_8)
    feat_32 = net.feat_32(feat_16)
    feat_64 = net.feat_64(feat_32)
    return feat_8, feat_16, feat_32, feat_64

def get_late_features(net, im_size, feat_64, feat_8, feat_16, feat_32):
    feat_128 = net.feat_128(feat_64)
    feat_128 = net.se_128(feat_8, feat_128)

    feat_256 = net.feat_256(feat_128)
    feat_256 = net.se_256(feat_16, feat_256)
    if im_size==256:
        return net.to_big(feat_256)
    
    feat_512 = net.feat_512(feat_256)
    feat_512 = net.se_512(feat_32, feat_512)
    if im_size==512:
        return net.to_big(feat_512)
    
    feat_1024 = net.feat_1024(feat_512)
    return net.to_big(feat_1024)


feat_8_a, feat_16_a, feat_32_a, feat_64_a = get_early_features(netG, noise_a)
feat_8_b, feat_16_b, feat_32_b, feat_64_b = get_early_features(netG, noise_b)

images_b = get_late_features(netG, im_size, feat_64_b, feat_8_b, feat_16_b, feat_32_b)
images_a = get_late_features(netG, im_size, feat_64_a, feat_8_a, feat_16_a, feat_32_a)

imgs = [ torch.ones(1, 3, im_size, im_size) ]
imgs.append(images_b.cpu())
for i in range(bs):
    imgs.append(images_a[i].unsqueeze(0).cpu())

    gimgs = get_late_features(netG, im_size, feat_64_a[i].unsqueeze(0).repeat(bs, 1, 1, 1), feat_8_b, feat_16_b, feat_32_b)
    imgs.append(gimgs.cpu())

imgs = torch.cat(imgs)
vutils.save_image(imgs.add(1).mul(0.5), 'style_mix_1.jpg', nrow=bs+1)