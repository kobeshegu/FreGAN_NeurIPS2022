import torch
from torch import nn, real, select
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator, SimpleDecoder
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        #pred, [rec_all, rec_small, rec_part], part = net(data, label)
        pred = net(data, label)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() #+ \
            #percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            #percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            #percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item()#, rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
@torch.no_grad()
def interpolate(z1, z2, netG, img_name, step=8):
    z = [  a*z2 + (1-a)*z1 for a in torch.linspace(0, 1, steps=step)  ]
    z = torch.cat(z).view(step, -1)
    g_image = netG(z)[0]
    vutils.save_image( g_image.add(1).mul(0.5), img_name , nrow=step)


def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
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
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=data_root, transform=trans, return_idx=True)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))

    total_iterations = int(len(dataset)*100/batch_size)
    
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)

    
    ckpt = torch.load(checkpoint)
    load_params( netG , ckpt['g_ema'] )
    #netG.eval()
    netG.to(device)

    fixed_noise = torch.randn(len(dataset), nz, requires_grad=True, device=device)
    optimizerG = optim.Adam([fixed_noise], lr=0.1, betas=(nbeta1, 0.999))

    log_rec_loss = 0


    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image, noise_idx = next(dataloader)
        real_image = real_image.to(device)

        optimizerG.zero_grad()
        
        select_noise = fixed_noise[noise_idx]
        g_image = netG(select_noise)[0]

        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() + 0.2*F.mse_loss(g_image, real_image)

        rec_loss.backward()

        optimizerG.step()

        log_rec_loss += rec_loss.item()

        if iteration % 100 == 0:
            print("lpips loss g: %.5f"%(log_rec_loss/100))
            log_rec_loss = 0

        if iteration % (save_interval*10) == 0:
            
            with torch.no_grad():
                vutils.save_image( torch.cat([
                        real_image, g_image]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration , nrow=batch_size)

                interpolate(fixed_noise[0], fixed_noise[1], netG, saved_image_folder+'/interpolate_0_1_%d.jpg'%iteration)
        
        if iteration % (save_interval*10) == 0 or iteration == total_iterations:
            torch.save(fixed_noise, saved_model_folder+'/%d.pth'%iteration)
            
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=True)
    
    mean_lpips = 0
    for idx, data in enumerate(dataloader):
        real_image, noise_idx = data
        real_image = real_image.to(device)

        select_noise = fixed_noise[noise_idx]
        g_image = netG(select_noise)[0]

        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() 
        mean_lpips += rec_loss.sum()
    mean_lpips /= len(dataset)
    print(mean_lpips)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=4, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')


    args = parser.parse_args()
    print(args)

    train(args)