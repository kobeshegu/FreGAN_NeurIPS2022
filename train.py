import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import numpy as np
import argparse
import random
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH

def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


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

class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr is None:
            decay = self.decay
        elif itr < self.start_itr:#if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))
      ##  return self.target_dict


def lecam_reg(dis_real, dis_fake, ema_real, ema_fake):
  reg = torch.mean(F.relu(dis_real - ema_fake).pow(2)) + \
        torch.mean(F.relu(ema_real - dis_fake).pow(2))
  return reg

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part], skips = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        # err.backward()
        #  pred.mean().item(), rec_all, rec_small, rec_part
        return err, rec_all, rec_small, rec_part, skips
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        #err.backward()
        #return pred.mean().item()
        return err
        

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
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder, saved_freimage_folder = get_dir(args)
    start_ema = 0
    tb_writer = SummaryWriter("./train_results/Panda_FreGAN/logs")
    vision_tag = ['D-loss', 'D-loss-real', 'D-loss-fake', 'G-loss']

    
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
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)
    #print(netG)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)
    #print(netD)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    toPIL = transforms.ToPILImage()
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    Pool = WavePool(3).cuda()

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    # D_ema = netD
    # ema_class = ema(netD, D_ema, decay=0.9999, start_itr=start_ema)

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)


        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        criteria = torch.nn.L1Loss()

        fake_images, _, _, _ = netG(noise, skips=None)

        #real_image_vis = real_image

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part, skips = train_d(netD, real_image, label="real")
#        ema_class.update(current_iteration)
        # for key, value in ema1.items():
        #      print(key)
            # print(value)
        # ema_fake = D_ema(fake_images, "fake")
        # ema_real, rec_img_all, rec_img_small, rec_img_part, skips = train_d(netD, real_image, label="real")
        err_df = train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        # lecam_reg_value = lecam_reg(err_df, err_dr, ema_real.detach(), ema_fake.detach())
        loss = err_df + err_dr #+ lecam_reg_value
        loss.backward()
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        fake_images2, fres_8, fres_16, fres_32 = netG(noise, skips=None)
        fake_images2 = [DiffAugment(fake, policy=policy) for fake in fake_images2]
        real_fre32 = skips['conv2_1'].clone().detach()
        real_fre16 = skips['conv3_1'].clone().detach()
        real_fre8 = skips['conv4_1'].clone().detach()
        loss_feat_32 = criteria(real_fre32, fres_32)
        loss_feat_16 = criteria(real_fre16, fres_16)
        loss_feat_8 = criteria(real_fre8, fres_8)
        pred_g = netD(fake_images2, "fake")
        err_g = -pred_g.mean() + loss_feat_8 + loss_feat_16 + loss_feat_32

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 1000 == 0:
            print(real_image.size())
            print(fake_images2[0].size())
            LL_real, LH_real, HL_real, HH_real = Pool(real_image_vis)
            print(LL_real.size())
            LL_fake, LH_fake, HL_fake, HH_fake = Pool(fake_images2[0])
            for i in range(8):
                img_real = real_image_2[i, :, :, :]
               # print(img_real.size())
                img_real = toPIL(img_real)
                img_real.save(os.path.join(saved_freimage_folder, 'img_real_{}_{}.png'.format(iteration, i)),  'png')

                fake_image = fake_images2[0]
                img_fake = fake_image[i, :, :, :]
                img_fake = toPIL(img_fake)
                img_fake.save(os.path.join(saved_freimage_folder, 'img_fake_{}_{}.png'.format(iteration, i)),  'png')

                realLL = LL_real[i, :, :, :]
#                print(img_realLL.size())
                img_realLL = toPIL(realLL)
                img_realLL.save(os.path.join(saved_freimage_folder, 'LL_real_{}_Frequency{}.png'.format(iteration, i)), 'png')
                realLH = LH_real[i, :, :, :]
                img_realLH = toPIL(realLH)
                img_realLH.save(os.path.join(saved_freimage_folder, 'LH_real_{}_Frequency{}.png'.format(iteration, i)), 'png')
                realHL = HL_real[i, :, :, :]
                img_realHL = toPIL(realHL)
                img_realHL.save(os.path.join(saved_freimage_folder, 'HL_real_{}_Frequency{}.png'.format(iteration, i)), 'png')
                realHH = HH_real[i, :, :, :]
                img_realHH = toPIL(realHH)
                img_realHH.save(os.path.join(saved_freimage_folder, 'HH_real_{}_Frequency{}.png'.format(iteration, i)), 'png')
                realHF = realLH + realHL + realHH
                img_realHF = toPIL(realHF)
                img_realHF.save(os.path.join(saved_freimage_folder, 'HF_real_{}_Frequency{}.png'.format(iteration, i)), 'png')

                fakeLL = LL_fake[i, :, :, :]
                img_fakeLL = toPIL(fakeLL)
                img_fakeLL.save(os.path.join(saved_freimage_folder, 'LL_fake_{}_Frequency{}.png'.format(iteration, i)), 'png')
                fakeLH = LH_fake[i, :, :, :]
                img_fakeLH = toPIL(fakeLH)
                img_fakeLH.save(os.path.join(saved_freimage_folder, 'LH_fake_{}_Frequency{}.png'.format(iteration, i)), 'png')
                fakeHL = HL_fake[i, :, :, :]
                img_fakeHL = toPIL(fakeHL)
                img_fakeHL.save(os.path.join(saved_freimage_folder, 'HL_fake_{}_Frequency{}.png'.format(iteration, i)),  'png')
                fakeHH = HH_fake[i, :, :, :]
                img_fakeHH = toPIL(fakeHH)
                img_fakeHH.save(os.path.join(saved_freimage_folder, 'HH_fake_{}_Frequency{}.png'.format(iteration, i)), 'png')
                fakeHF = fakeLH + fakeHL + fakeHH
                img_fakeHF = toPIL(fakeHF)
                img_fakeHF.save(os.path.join(saved_freimage_folder, 'HF_fake_{}_Frequency{}.png'.format(iteration, i)), 'png')

        for tag, value in zip(vision_tag, [err_df + err_dr,err_df, err_dr, err_g]):
            tb_writer.add_scalars(tag, {'train':value}, iteration)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (save_interval*10) == 0 and iteration !=0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise, skips=True)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128),
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)
        
        if (iteration % (save_interval*50)==0 and iteration!=0) or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./datasets/100-shot-panda/img', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='Panda_FreGAN', help='experiment name')
    parser.add_argument('--iter', type=int, default=100000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')

    args = parser.parse_args()
    print(args)

    train(args)
