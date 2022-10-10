import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy

import pdb


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def get_pred(x, resize, up, inception_model):
    if resize:
        x = up(x)
    x = inception_model(x)
    return F.softmax(x, dim=1).data.cpu().numpy()


def inception_score(imgs, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up device
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch_size_i = batch.size()[0]

        with torch.no_grad():
            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch.to(device), resize, up, inception_model)

        print(i+1, "/", len(dataloader), end="\r")

    print("\nComputing KL-Div Mean...")

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

        print(k+1, "/", splits, end="\r")
    print()
    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  #  parser.add_argument("--dataroot", type=str, default="./datasets/shells", help="Root directory for dataset")
    opt = parser.parse_args()
    output_name = '../train_results/Panda_FreGAN/IS.txt'
    f = open(output_name, 'w')
    for i in range(1,11):
        opt.dataroot = "../train_results/Panda_FreGAN/eval_" + str(i * 10000)

        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize((int(512),int(512))),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        print("Calculating Inception Score of iteration %d..."%i)
       #print(inception_score(IgnoreLabelDataset(dataset), batch_size=32, resize=True, splits=10))
        mean, std = inception_score(IgnoreLabelDataset(dataset), batch_size=32, resize=True, splits=10)
        print("Mean:%.10f"%mean)
        print("std:%.10f"%std)
        f.writelines('%s:%d\n' % ("Calculating Inception Score of iteration...",i))
        f.writelines('%s:%.10f\n' % ("Mean IS:", mean ))
        f.writelines('%s:%.10f\n' % ("Mean IS:", std))
        f.writelines('%s\n' % ("--------------------the is the split line---------------------"))
    f.close()

