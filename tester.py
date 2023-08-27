import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import utils
import dataset
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from itertools import cycle

def tester(opt):
    cudnn.benchmark = opt.cudnn_benchmark
    generator = utils.create_generator(opt)
    generator.eval()
    testset = dataset.InpaintDataset(opt, training=True)
    dataloader = DataLoader(testset)
    for id, (img, img_grey, mask, edge) in enumerate(dataloader):
        mask_write = mask.clone()
        img_write = img.clone()
        _, result = generator(img,mask,edge)
        result = img * (1 - mask) + result * mask
        result = result * 255
        mask_write = mask_write * 255
        img_write = img_write * 255
        result = result.type(torch.uint8)
        result = result.data[0, :, :, :].squeeze()
        img_write = img_write.type(torch.uint8)
        img_write = img_write.data[0,:,:,:].squeeze()
        mask_write = mask_write.type(torch.uint8)
        mask_write = torch.squeeze(mask_write.data[0, :, :, :], dim=1)
        torchvision.io.write_png(result, './output%d.png' % id)
        torchvision.io.write_png(mask_write, './mask%d.png' % id)
        torchvision.io.write_png(img_write, './img%d.png' % id)
