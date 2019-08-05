import sys
sys.path.append('utils')
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import densenet161 as densnet161
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as var
from PIL import Image as pilImg
import torch.nn.init as init
import torch.nn.functional as func
import random
import calcPSNR
import math
import torch.nn.init as init
import cv2
import sepconv
from argparse import ArgumentParser
import time
from tensorboardX import SummaryWriter

from network.FKCNN import Network
from utils.RGB_Dataset import mydataset

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data,0.0)

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    # ------------- Part for tensorboard --------------
    writer = SummaryWriter(comment="anchorFKCNN")
    # ------------- Part for tensorboard --------------
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(gpu)

    BATCH_SIZE=batch_size
    EPOCH=epoch

    LEARNING_RATE = lr
    belta1 = 0.9
    belta2 = 0.999

    trainset = mydataset(train_set,transform_train)
    valset = mydataset(valid_set)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)


    SepConvNet = Network().cuda()
    SepConvNet.apply(weights_init)
    # SepConvNet.load_state_dict(torch.load('/mnt/hdd/xiasifeng/sepconv/sepconv_mutiscale_LD/SepConv_iter33-ltype_fSATD_fs-lr_0.001-trainloss_0.1497-evalloss_0.1357-evalpsnr_29.6497.pkl'))

    # SepConvNet_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-5)

    # ----------------  Time part -------------------
    start_time = time.time()
    global_step = 0
    # ----------------  Time part -------------------

    for epoch in range(0,EPOCH):
        SepConvNet.train().cuda()
        cnt = 0
        sumloss = 0.0 # The sumloss is for the whole training_set
        tsumloss = 0.0 # The tsumloss is for the printinterval
        printinterval = 300
        print("---------------[Epoch%3d]---------------"%(epoch + 1))
        for imgL, imgR, label in trainLoader:
            global_step = global_step + 1
            cnt = cnt + 1
            SepConvNet_optimizer.zero_grad()
            imgL = var(imgL).cuda()
            imgR = var(imgR).cuda()
            label = var(label).cuda()
            output,output_a,output_b = SepConvNet(imgL, imgR)
            loss = 0.5*SepConvNet_cost(output, label) +\
            0.2*SepConvNet_cost(output_a, func.upsample(label,size=(label.shape[2]//4,label.shape[3]//4),mode='bilinear',align_corners=True)) +\
            0.3*SepConvNet_cost(output_b, func.upsample(label,size=(label.shape[2]//2,label.shape[3]//2),mode='bilinear',align_corners=True))
            loss.backward()
            SepConvNet_optimizer.step()
            sumloss = sumloss + loss.data.item()
            tsumloss = tsumloss + loss.data.item()
            if cnt % printinterval == 0:
                writer.add_image("Prev image", imgR[0], cnt)
                writer.add_image("Pred image", output[0], cnt)
                writer.add_scalar('Train Batch SATD loss', loss.data.item(), int(global_step / printinterval))
                writer.add_scalar('Train Interval SATD loss', tsumloss / printinterval, int(global_step / printinterval))
                print('Epoch [%d/%d], Iter [%d/%d], Time [%4.4f], Batch loss [%.6f], Interval loss [%.6f]' %
                    (epoch + 1, EPOCH, cnt, len(trainset) // BATCH_SIZE, time.time() - start_time, loss.data.item(), tsumloss / printinterval))
                tsumloss = 0.0
        print('Epoch [%d/%d], iter: %d, Time [%4.4f], Avg Loss [%.6f]' %
            (epoch + 1, EPOCH, cnt, time.time() - start_time, sumloss / cnt))

        # ---------------- Part for validation ----------------
        trainloss = sumloss / cnt
        SepConvNet.eval().cuda()
        evalcnt = 0
        pos = 0.0
        sumloss = 0.0
        psnr = 0.0
        for imgL, imgR, label in valLoader:
            imgL = var(imgL).cuda()
            imgR = var(imgR).cuda()
            label = var(label).cuda()
            with torch.no_grad():
                output, output_a, output_b = SepConvNet(imgL, imgR)
                loss = 0.5*SepConvNet_cost(output, label) + \
                    0.2*SepConvNet_cost(output_a,func.upsample(label, size=(label.shape[2] // 4, label.shape[3] // 4), mode='bilinear',align_corners=True)) + \
                    0.3*SepConvNet_cost(output_b,func.upsample(label, size=(label.shape[2] // 2, label.shape[3] // 2), mode='bilinear',align_corners=True))
                sumloss = sumloss + loss.data.item()
                psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                evalcnt = evalcnt + 1
        # ------------- Tensorboard part -------------
        writer.add_scalar("Valid SATD loss", sumloss / evalcnt, epoch)
        writer.add_scalar("Valid PSNR", psnr / valset.__len__(), epoch)
        # ------------- Tensorboard part -------------
        print('Validation loss [%.6f],  Average PSNR [%.4f]' % (
        sumloss / evalcnt, psnr / valset.__len__()))
        SepConvNet_schedule.step(psnr / valset.__len__())
        torch.save(SepConvNet.state_dict(),
                os.path.join('.', 'anchor_SepConv_iter' + str(epoch + 1)
                                + '-ltype_fSATD_fs'
                                + '-lr_' + str(LEARNING_RATE)
                                + '-trainloss_' + str(round(trainloss, 4))
                                + '-evalloss_' + str(round(sumloss / evalcnt, 4))
                                + '-evalpsnr_' + str(round(psnr / valset.__len__(), 4)) + '.pkl'))
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=16, help="Mini-batch size")
    parser.add_argument("--epoch", type=int, dest="epoch",
                        default=200, help="Number of epochs")
    parser.add_argument("--gpu", type=int, dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--train_set", type=str, dest="train_set", default="/mnt/ssd/iku/vimeoimgs_LD_crop", 
                        help="Path of the training set")
    parser.add_argument("--valid_set", type=str, dest="valid_set", default="/mnt/hdd/iku/vimeoimgs_LD_val_crop", 
                        help="Path of the validation set")

    args = parser.parse_args()
    main(**vars(args))
