import os
import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import networks


def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


def create_generator(opt):
    # Initialize the networks
    generator = networks.GatedGenerator(opt)
    print('Generator is created!')
    if opt.load_name_g:
        generator.load_state_dict(torch.load(opt.load_name_g))
        print('Load generator %s' % opt.load_name_g)
    else:
        # Init the networks
        weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator


def create_discriminator(opt):
    # Initialize the networks
    discriminator = networks.PatchDiscriminator(opt)
    print('Discriminator is created!')
    if opt.load_name_d:
        discriminator.load_state_dict(torch.load(opt.load_name_d))
        print('Load generator %s' % opt.load_name_d)
    else:
        weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator


def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = networks.PerceptualNet()
    # Pre-trained VGG-16
    try:
        vgg16 = torch.load('./vgg16_pretrained.pth')
    except:
        vgg16 = torchvision.models.vgg16(pretrained=True)
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret


