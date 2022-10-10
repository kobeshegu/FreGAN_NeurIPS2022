from easing_functions.easing import LinearInOut
import torch
import pandas as pd 
from torchvision import utils as vutils
import os
import cv2
from tqdm import tqdm
from scipy import io
import numpy as np
import argparse

from easing_functions import QuadEaseInOut
from easing_functions import SineEaseIn, SineEaseInOut, SineEaseOut
from easing_functions import ElasticEaseIn, ElasticEaseInOut, ElasticEaseOut

ease_fn_dict = {'QuadEaseInOut': QuadEaseInOut,
                'SineEaseIn': SineEaseIn, 
                'SineEaseInOut': SineEaseInOut, 
                'SineEaseOut': SineEaseOut,
                'ElasticEaseIn': ElasticEaseIn, 
                'ElasticEaseInOut': ElasticEaseInOut, 
                'ElasticEaseOut': ElasticEaseOut,
                'Linear': LinearInOut}

def interpolate(z1, z2, num_interp):
    # this is a "first frame included, last frame excluded" interpolation
    w = torch.linspace(0, 1, num_interp+1)
    interp_zs = []
    for n in range(num_interp):
        interp_zs.append( (z2*w[n].item() + z1*(1-w[n].item())).unsqueeze(0) )
    return torch.cat(interp_zs)



def interpolate_ease_inout(z1, z2, num_interp, ease_fn, model_type='freeform'):
    # this is a "first frame included, last frame excluded" interpolation
    w = ease_fn(start=0, end=1, duration=num_interp+1)
    interp_zs = []

    # just to make sure the latent vectors in the right shape
    if model_type == 'freeform':
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
    if model_type == 'stylegan2':
        if type(z1) is list:
            z1 = [z1[0].view(1, -1), z1[1].view(1, -1)]
        else:
            z1 = [z1.view(1, -1), z1.view(1, -1)]
        if type(z2) is list:
            z2 = [z2[0].view(1, -1), z2[1].view(1, -1)]
        else:
            z2 = [z2.view(1, -1), z2.view(1, -1)]

    for n in range(num_interp):
        if model_type == 'freeform':
            interp_zs.append( z2*w.ease(n) + z1*(1-w.ease(n)) )
        if model_type == 'stylegan2':
            interp_zs.append( [ z2[0]*w.ease(n) + z1[0]*(1-w.ease(n)),
                                z2[1]*w.ease(n) + z1[1]*(1-w.ease(n)) ] )
    return interp_zs

@torch.no_grad()
def net_generate(netG, z, model_type='freeform', im_size=1024):
    
    if model_type == 'stylegan2':
        z_contents = []
        z_styles = []
        for zidx in range(len(z)):
            z_contents.append(z[zidx][0])
            z_styles.append(z[zidx][1])
        z = [ torch.cat(z_contents), torch.cat(z_styles) ]
        gimg = netG( z, inject_index=8, input_is_latent=True, randomize_noise=False )[0].cpu()
    elif model_type == 'freeform':
        z = torch.cat(z)
        gimg = netG(z)[0].cpu()

    return torch.nn.functional.interpolate(gimg, im_size)

def batch_generate_and_save(netG, zs, folder_name, batch_size=8, model_type='freeform', im_size=1024):
    # zs is a list of vectors if model is freeform
    # zs is a list of lists, each list is 2 vectors, if model is stylegan
    t = 0
    num = 0
    if len(zs) < batch_size:
        gimgs = net_generate(netG, zs, model_type, im_size=im_size).cpu()
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1

    for k in tqdm(range(len(zs)//batch_size)):
        gimgs = net_generate(netG, zs[k*batch_size:(k+1)*batch_size], model_type, im_size=im_size)
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1
        t = k

    if len(zs)%batch_size>0:
        gimgs = net_generate(netG, zs[(t+1)*batch_size:], model_type, im_size=im_size)        
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
            num += 1



def batch_save(images, folder_name, start_num=0):
    os.makedirs(folder_name, exist_ok=True)
    num = start_num
    for image in images:
        vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.jpg"%(num) )
        num += 1


def read_img_and_make_video(dist, video_name, fps):
    img_array = []
    for i in tqdm(range(len(os.listdir(dist)))):
        try:
            filename = dist+'/%d.jpg'%(i)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        except:
            print('error at: %d'%i)
    
    if '.mp4' not in video_name:
        video_name += '.mp4'
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

from shutil import rmtree

def make_video_from_latents(net, selected_latents, frames_dist_folder, video_name, fps, video_length, ease_fn, model_type, im_size=1024):
    # selected_latents: the latent noise of user selected key-frame images, it is a list
    # each item in the list is a vector if the model is freeform, 
    # each item in the list is a list of two vectors if the model is stylegan2
    # frames_dist_folder: the folder path to save the generated images to make the video
    # fps: is the frames we generate per second
    # video_length: is the time of the video, in seconds. For example: 30 means a video length of 30 seconds
    # ease_fn: user selected type of transitions between each key-frame

    # first calculate how many images need to generate
    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    nbr_generate = fps*video_length
    nbr_keyframe = len(selected_latents)
    nbr_interpolation = 1 + nbr_generate // (nbr_keyframe - 1)
    

    main_zs = []
    for idx in range(nbr_keyframe-1):
        main_zs += interpolate_ease_inout(selected_latents[idx], 
                            selected_latents[idx+1], nbr_interpolation, ease_fn, model_type)
    

    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=8, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)


if __name__ == "__main__":


    device = torch.device('cuda:%d'%(0))

    load_model_err = 0

    from models import Generator as Generator_freeform
    
    frames_dist_folder = 'project_video_frames' # a folder to save generated images
    ckpt_path = './time_1024_1/models/180000.pth' # path to the checkpoint
    video_name = 'videl_keyframe_15'  # name of the generated video

    model_type = 'freeform'
    net = Generator_freeform(ngf=64, nz=100)
    net.load_state_dict(torch.load(ckpt_path)['g'])
    net.to(device)
    net.eval()

    
    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    fps = 30
    minutes = 1
    im_size = 1024
    
    ease_fn=ease_fn_dict['SineEaseInOut']

    init_kf_nbr = 15
    nbr_key_frames_per_minute = [init_kf_nbr-i for i in range(minutes)]
    nbr_key_frames_total = sum(nbr_key_frames_per_minute)
    noises = torch.randn( nbr_key_frames_total , 100).to(device)
    user_selected_noises = [n for n in noises]
    nbr_interpolation_list = [[fps*60//nbr_kf]*nbr_kf for nbr_kf in nbr_key_frames_per_minute]
    nbl = []
    for nb in nbr_interpolation_list:
        nbl += nb

    print(len(nbl)) 
    print(len(user_selected_noises))# , print("mismatch size")
    main_zs = []
    for idx in range(len(user_selected_noises)-1):
        main_zs += interpolate_ease_inout(user_selected_noises[idx], 
                            user_selected_noises[idx+1], nbl[idx], ease_fn, model_type)
    for idx in range(100):
        main_zs.append(main_zs[-1])
    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=8, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)

