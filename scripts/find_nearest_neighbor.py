from eval import load_params
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision import transforms
import os
import random
import argparse
from tqdm import tqdm

from models import Generator
from operation import load_params, InfiniteSamplerWrapper

noise_dim = 256
device = torch.device('cuda:%d'%(0))

im_size = 512  
net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=im_size)#, big=args.big )
net_ig.to(device)

epoch = 50000
ckpt = './models/all_%d.pth'%(epoch)
checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
net_ig.load_state_dict(checkpoint['g'])
load_params(net_ig, checkpoint['g_ema'])

batch = 8
noise = torch.randn(batch, noise_dim).to(device)
g_imgs = net_ig(noise)[0]

vutils.save_image(g_imgs.add(1).mul(0.5), 
                    os.path.join('./', '%d.png'%(2)))


transform_list = [
            transforms.Resize((int(256),int(256))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
trans = transforms.Compose(transform_list)
data_root = '/media/database/images/first_1k'
dataset = ImageFolder(root=data_root, transform=trans)

import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

the_image = g_imgs[0].unsqueeze(0)
def find_closest(the_image):
    the_image = F.interpolate(the_image, size=256)
    small = 100
    close_image = None
    for i in tqdm(range(len(dataset))):
        real_iamge = dataset[i][0].unsqueeze(0).to(device)

        dis = percept(the_image, real_iamge).sum()
        if dis < small:
            small = dis
            close_image = real_iamge
    return close_image, small

all_dist = []
batch = 8
result_path = 'nn_track'
import os
os.makedirs(result_path, exist_ok=True)
for j in range(8):
    with torch.no_grad():
        noise = torch.randn(batch, noise_dim).to(device)
        g_imgs = net_ig(noise)[0]

    for n in range(batch):
        the_image = g_imgs[n].unsqueeze(0)

        close_0, dis = find_closest(the_image)
        
        vutils.save_image(torch.cat([F.interpolate(the_image,256), close_0]).add(1).mul(0.5), \
            result_path+'/nn_%d.jpg'%(j*batch+n))
        all_dist.append(dis.view(1))

new_all_dist = []
for v in all_dist:
    new_all_dist.append(v.view(1))
print(torch.cat(new_all_dist).mean())