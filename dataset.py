import os
import cv2
import torch
from PIL import Image
import numpy as np
import utils
from torchvision import transforms
from torch.utils.data import Dataset

class InpaintDataset(Dataset):
    def __init__(self, opt, training):
        self.opt = opt
        self.imglist = utils.get_files(opt.datasetroot)
        self.training = training
        self.length = len(self.imglist)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        img = Image.open(self.getImageDir(index)).resize([self.opt.imgsize, self.opt.imgsize]).convert("RGB")
        mask = self.generate_stroke_mask([self.opt.imgsize, self.opt.imgsize], maxBrushWidth=40)
        img_grey = img.convert('L')
        edge = self.load_edge(img_grey, mask)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        trans2 = transforms.Compose([transforms.ToTensor(), ])
        img = trans2(img)
        img_grey = trans2(img_grey)
        edge = trans2(edge)
        mask = trans2(mask)
        return img, img_grey, mask, edge

    def getImageDir(self, index):
        return os.path.join(self.opt.datasetroot, self.imglist[index])

    def generate_stroke_mask(self, im_size, parts=7, maxVertex=25, maxLength=80, maxBrushWidth=80, maxAngle=360):
        mask = np.zeros((im_size[0], im_size[1]), dtype=np.float32)

        def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
            mask = np.zeros((h, w), np.float32)
            numVertex = np.random.randint(maxVertex + 1)
            startY = np.random.randint(h)
            startX = np.random.randint(w)
            brushWidth = 0
            for i in range(numVertex):
                angle = np.random.randint(maxAngle + 1)
                angle = angle / 360.0 * 2 * np.pi
                if i % 2 == 0:
                    angle = 2 * np.pi - angle
                length = np.random.randint(maxLength + 1)
                brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
                nextY = startY + length * np.cos(angle)
                nextX = startX + length * np.sin(angle)
                nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
                nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
                cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
                cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
                startY, startX = nextY, nextX
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            return mask

        for i in range(parts):
            mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)
        return Image.fromarray(mask)

    def load_edge(self, img, mask):
        img = np.asarray(img)
        mask = np.asarray(mask)
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        edge = cv2.Canny(img, 50, 200, apertureSize=3).astype(np.bool)
        if mask is not None:
            return Image.fromarray(np.multiply(edge, mask))
        return Image.fromarray(edge)
