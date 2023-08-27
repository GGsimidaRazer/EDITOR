import cv2
import argparse
from torch.utils.data import DataLoader
from dataset import InpaintDataset
import modules
import networks
import torch.nn.functional as F
import torch
import utils
from itertools import cycle
import numpy
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetroot", type=str, default="/home/jason/Data/Places2/data_large/a/airfield")
    parser.add_argument("--imgsize", type=int, default=512)
    parser.add_argument("--activation",type=str, default="elu")
    parser.add_argument("--norm", type=str, default="none")
    parser.add_argument('--batch_size',type=str,default=1)
    parser.add_argument('--lambda_l1', type = float, default = 0.256, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 0.1, help = 'the parameter of FML1Loss (perceptual loss)')

    opt = parser.parse_args()
    print(opt)
    set = InpaintDataset(opt, training=False)
    load = DataLoader(set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    L1Loss = torch.nn.L1Loss()
    RELU = torch.nn.ReLU()
    model = networks.GatedGenerator(opt).cuda()
    perceptualnet = utils.create_perceptualnet().cuda()
    discriminator = networks.PatchDiscriminator(opt).cuda()
    optimizer_g1 = torch.optim.Adam(model.coarse.parameters(), lr=4e-4)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=4e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=4e-4)
    it = iter(cycle(load))
    for id, (img, img_grey, mask, edge) in enumerate(load):
        '''
        img = img.cuda()
        mask = mask.cuda()
        edge = edge.cuda()
        utils.weights_init(model)
        first_out, second_out = model(img,mask,edge)
        first_out_wholeimg = img * (1 - mask) + first_out * mask
        second_out_wholeimg = img * (1 - mask) + second_out * mask

        optimizer_d.zero_grad()
        fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
        true_scalar = discriminator(img, mask)
        # W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+ gradient_penalty(discriminator, img, second_out_wholeimg, mask)
        hinge_loss = torch.mean(RELU(1 - true_scalar)) + torch.mean(RELU(fake_scalar + 1))
        loss_D = hinge_loss
        loss_D.backward(retain_graph=True)
        optimizer_d.step()


        first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
        second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
        fake_scalar = discriminator(second_out_wholeimg, mask)
        GAN_Loss = - torch.mean(fake_scalar)

        optimizer_g1.zero_grad()
        first_MaskL1Loss.backward(retain_graph=True)
        optimizer_g1.step()

        optimizer_g.zero_grad()
        img_featuremaps = perceptualnet(img)  # feature maps
        second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
        second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

        #loss = 0.5 * opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + GAN_Loss + second_PerceptualLoss * opt.lambda_perceptual
        loss = 0.5 * opt.lambda_l1 * L1Loss(first_out_wholeimg, img)
        loss.backward()
        optimizer_g.step()
        '''''
        for i in range(5):
            img1, img_grey1, mask1, edge1 = next(it)
            if numpy.asarray(img1).any() == numpy.asarray(img).any():
                print('True')

        print("id:%d" % id)


