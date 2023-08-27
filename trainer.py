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



def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark
    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    #dist_file = "file://" + opt.dist_url + "dist_file.{}".format(str(opt.job_id))
    #dist.init_process_group(backend='nccl', init_method=dist_file, world_size=opt.world_size, rank=opt.rank)

    #torch.cuda.set_device(opt.local_rank)

    # Build networks
    generator = utils.create_generator(opt).cuda()
    discriminator = utils.create_discriminator(opt).cuda()
    perceptualnet = utils.create_perceptualnet().cuda()

    # To device

    # edgeGenerator = edgeGenerator.cuda()
    # edgeDiscriminator = edgeDiscriminator.cuda()


    # Loss functions
    L1Loss = nn.L1Loss()  # reduce=False, size_average=False)
    RELU = nn.ReLU()
    # edgeL1Loss = nn.L1Loss()
    # edgeLoss = AdversarialLoss().cuda()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d)

    # optimizer_ge = torch.optim.Adam(edgeGenerator.parameters(), lr=opt.lr_ge, betas=(0.0,0.9))
    # optimizer_de = torch.optim.Adam(edgeDiscriminator.parameters(), lr=opt.lr_de, betas=(0.0,0.9))

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Save the model if pre_train == True
    def save_model(net, epoch, opt, batch=0, is_D=False):
        """Save the model at "checkpoint_interval" and its multiple"""
        if is_D:
            #model_name = 'discriminator_WGAN_epoch%d_batch%d.pth' % (epoch + 1, batch)
            model_name = opt.datasetname + '_D_epoch%d_batch%d.pth' % (epoch, batch)
        else:
            #model_name = 'deepfillv2_WGAN_epoch%d_batch%d.pth' % (epoch + 1, batch)
            model_name = opt.datasetname + '_G_epoch%d_batch%d.pth' % (epoch, batch)
        model_name = os.path.join(save_folder, model_name)
        # if opt.multi_gpu:
        #    if epoch % opt.checkpoint_interval == 0:
        #        torch.save(net.module.state_dict(), model_name)
        #        print('The trained model is successfully saved at epoch %d batch %d' % (epoch, batch))
        # else:
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d batch %d' % (epoch, batch))

    def loop_dataloader(dataloader):
        while True:
            for batch_idx, (img, img_grey, mask, edge) in enumerate(dataloader):
                yield img, img_grey, mask, edge


    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt, training=True)
    print('The overall number of images equals to %d' % len(trainset))


    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers)

    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    # Training loop
    optimizer_d.zero_grad()
    optimizer_g.zero_grad()
    length = len(dataloader)
    total = length // opt.accumulation_step
    interval = total // 100
    model_inverval = total // 10
    dataloader = loop_dataloader(dataloader)
    for epoch in range(opt.epochs):
        writer = SummaryWriter(log_dir='runs/'+opt.datasetname+'/%d' % epoch)
        print("Start epoch ", epoch + 1, "!")
        pbar = tqdm(total=total)
        for batch_idx in range(length):
            step = (batch_idx + 1) // opt.accumulation_step
            for rep in range(opt.repeat_D):
                img1, img_grey1, mask1, edge1 = next(dataloader)
                img1 = img1.cuda()
                mask1 = mask1.cuda()
                edge1 = edge1.cuda()
                _, second_out1 = generator(img1, mask1, edge1)
                second_out_wholeimg1 = img1 * (1 - mask1) + second_out1 * mask1  # in range [0, 1]
                fake_scalar1 = discriminator(second_out_wholeimg1, mask1)
                true_scalar1 = discriminator(img1, mask1)
            #    # W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+ gradient_penalty(discriminator, img, second_out_wholeimg, mask)
                #hinge_loss = torch.mean(RELU(1 - true_scalar)) + torch.mean(RELU(fake_scalar + 1))
                loss_D = torch.mean(RELU(1 - true_scalar1)) + torch.mean(RELU(fake_scalar1 + 1))
                loss_D = loss_D / opt.accumulation_step
                loss_D.backward()

            #for wk in range(1):
            #    fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            #    true_scalar = discriminator(img, mask)
            #    # W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+ gradient_penalty(discriminator, img, second_out_wholeimg, mask)
            #    hinge_loss = torch.mean(RELU(1 - true_scalar)) + torch.mean(RELU(fake_scalar + 1))
            #    loss_D = hinge_loss
            #    loss_D.backward(retain_graph=True)


            if (batch_idx + 1) % opt.accumulation_step == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img, img_grey, mask, edge = next(dataloader)
            img = img.cuda()
            mask = mask.cuda()
            edge = edge.cuda()

            # forward propagation
            first_out, second_out = generator(img, mask, edge)
            first_out_wholeimg = img * (1 - mask) + first_out * mask

            second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

            ### Train Generator

            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = - torch.mean(fake_scalar)

            # optimizer_g1.zero_grad()
            # first_MaskL1Loss.backward(retain_graph=True)
            # optimizer_g1.step()



            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)  # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + GAN_Loss + second_PerceptualLoss * opt.lambda_perceptual
            #loss = first_MaskL1Loss + 1.2*second_MaskL1Loss + 1e-4*GAN_Loss + second_PerceptualLoss * opt.lambda_perceptual
            loss = loss / opt.accumulation_step
            loss.backward()

            if (batch_idx + 1) % opt.accumulation_step == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                writer.add_scalar('first_MaskL1Loss', first_MaskL1Loss.item(), step)
                writer.add_scalar('second_MaskL1Loss', second_MaskL1Loss.item(), step)
                writer.add_scalar('G_Loss', GAN_Loss.item(), step)
                writer.add_scalar('D_Loss', loss_D.item(), step)
                writer.add_scalar('Perceptual_Loss', second_PerceptualLoss.item(), step)
                writer.flush()

            # Determine approximate time left
            #batches_done = epoch * len(dataloader) + batch_idx
            #batches_left = opt.epochs * len(dataloader) - batches_done



            # Print log
            # print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
            #      ((epoch + 1), opt.epochs, (batch_idx+1), len(dataloader), first_MaskL1Loss.item(),
            #       second_MaskL1Loss.item()))
            # print("\r[D Loss: %.5f] [Perceptual Loss: %.5f] [G Loss: %.5f] time_left: %s" %
            #      (loss_D.item(), second_PerceptualLoss.item(), GAN_Loss.item(), time_left))

                pbar.update(1)

            if step % interval == 0:
                # Generate Visualization image

                img, mask, first_out, second_out, first_out_wholeimg, second_out_wholeimg = img.cpu(), mask.cpu(), first_out.cpu(), second_out.cpu(), \
                                                                                     first_out_wholeimg.cpu(), second_out_wholeimg.cpu()
                '''
                img = (img+1)/2
                first_out = (first_out+1)/2
                second_out = (second_out+1)/2
                first_out_wholeimg = (first_out_wholeimg+1)/2
                second_out_wholeimg = (second_out_wholeimg+1)/2
                '''

                masked_img = img * (1 - mask) + mask
                img_save = torch.cat((img, masked_img, first_out, second_out, first_out_wholeimg, second_out_wholeimg),
                                     3)
                # Recover normalization: * 255 because last layer is sigmoid activated
                #img_save = F.interpolate(img_save, scale_factor=0.5, recompute_scale_factor=True)

                img_copy = img_save.clone()
                img_copy = img_copy * 255

                # Process img_copy and do not destroy the data of img
                img_copy = img_copy.type(torch.uint8)
                save_img_name = 'sample_batch' + str(step) + '.png'
                save_img_path = os.path.join(sample_folder, save_img_name)
                writer.add_images('epoch_%d_sample_batch_%d' % (epoch, step), img_copy, epoch, dataformats='NCHW')
                writer.flush()
                torchvision.io.write_png(img_copy.data[0,:,:,:].squeeze(), save_img_path)
                '''
                img_copy = img_save.clone().data.permute(0, 2, 3, 1)[0, :, :, :].numpy()
                # img_copy = np.clip(img_copy, 0, 255)
                img_copy = img_copy.astype(np.uint8)
                save_img_name = 'sample_batch' + str(step) + '.png'
                save_img_path = os.path.join(sample_folder, save_img_name)
                writer.add_images('epoch_%d_sample_batch_%d' % (epoch, step), img_copy, epoch, dataformats='HWC')
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, img_copy)
                '''


            if step % model_inverval == 0:
                save_model(generator, epoch, opt, step)
                save_model(discriminator, epoch, opt, step, is_D=True)

            # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

            # Save the model
        save_model(generator, epoch+1, opt)
        save_model(discriminator, epoch+1, opt, is_D=True)

        pbar.close()
        writer.close()