'''
This version is the raw implementation of FKCNN without multi-scale
'''
import sys
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
import IPython

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

class vimeodataset(data.Dataset):
    def __init__(self, datasetpath, filelist, transform=None):
        self.transform=transform
        self.readdir = datasetpath
        self.imgstrs = []
        for filename in os.listdir(self.readdir):
            if filename.__len__() > 7 and filename[-7:] == '_1R.bmp':
                self.imgstrs.append(filename)
        # self.imgstrs = self.imgstrs[:122]

    def __len__(self):
        return len(self.imgstrs)

    def __getitem__(self, idx):
        imgstr = self.imgstrs[idx]
        
        raw_imgs = []
        drop_imgs = []
        for i in range(7):
            fname = os.path.join(self.readdir, imgstr[:-7] + '_%dR.bmp'%(i + 1))
            raw_imgs.append(np.array(pilImg.open(fname)))
            fname = os.path.join(self.readdir, imgstr[:-7] + '_%dC.bmp'%(i + 1))
            drop_imgs.append(np.array(pilImg.open(fname)))
        
        if self.transform is not None:
            # random shift and crop
            cropx = random.randint(2, 20)
            cropy = random.randint(2, 20)

            for i in range(7):
                raw_imgs[i] = raw_imgs[i][cropy:cropy + 128, cropx:cropx + 128, :]
                drop_imgs[i] = drop_imgs[i][cropy:cropy + 128, cropx:cropx + 128, :]

            flipH = random.randint(0, 1)
            flipV = random.randint(0,1)
            if flipH==1:
                for i in range(7):
                    raw_imgs[i] = np.flip(raw_imgs[i],0)
                    drop_imgs[i] = np.flip(drop_imgs[i],0)

            if flipV==1:
                for i in range(7):
                    raw_imgs[i] = np.flip(raw_imgs[i],1)
                    drop_imgs[i] = np.flip(drop_imgs[i],1)
        raw_imgs.extend(drop_imgs)
        # IPython.embed()
        # exit()
        try:
            ret =  [var(torch.from_numpy(tmpimg.transpose(2,0,1).astype(np.float32) / 255.0)) for tmpimg in raw_imgs]
            return ret
        except Exception:
            IPython.embed()
            exit()


class mydataset(data.Dataset):
    def __init__(self, datasetpath, transform=None):
        self.transform=transform
        self.readdir = datasetpath
        self.imgstrs = []
        for filename in os.listdir(self.readdir):
            if filename.__len__() > 6 and filename[-6:] == '_L.bmp':
                self.imgstrs.append(filename)

    def __len__(self):
        return len(self.imgstrs)

    def __getitem__(self, idx):
        imgstr = self.imgstrs[idx]
        img1str = os.path.join(self.readdir, imgstr)
        img2str = os.path.join(self.readdir, imgstr[:-6] + '_R.bmp')

        corder = 0
        if corder == 0 or self.transform is None:
            img1 = np.array(pilImg.open(img1str))
            img2 = np.array(pilImg.open(img2str))
        else:
            img1 = np.array(pilImg.open(img2str))
            img2 = np.array(pilImg.open(img1str))

        img3 = np.array(pilImg.open(os.path.join(self.readdir, imgstr[:-6] + '_M.bmp')))

        if self.transform is not None:
            # random shift and crop
            cropx = random.randint(2, 20)
            cropy = random.randint(2, 20)



            shift = random.randint(0, 2)
            ifx = random.randint(0, 1)
            shiftx = 0
            shifty = 0

            img3 = img3[cropy:cropy + 128, cropx:cropx + 128, :]
            img1 = img1[cropy - shifty:cropy + 128 - shifty, cropx - shiftx : cropx + 128 - shiftx,:]
            img2 = img2[cropy + shifty:cropy + 128 + shifty, cropx + shiftx : cropx + 128 + shiftx,:]

            flipH = random.randint(0, 1)
            flipV = random.randint(0,1)
            if flipH==1:
                img1=np.flip(img1,0)
                img2 = np.flip(img2, 0)
                img3 = np.flip(img3, 0)
            if flipV==1:
                img1=np.flip(img1,1)
                img2 = np.flip(img2, 1)
                img3 = np.flip(img3, 1)

        return var(torch.from_numpy(img1.transpose(2,0,1).astype(np.float32) / 255.0)), var(torch.from_numpy(img2.transpose(2,0,1).astype(np.float32) / 255.0)), \
               var(torch.from_numpy(img3.transpose(2,0,1).astype(np.float32) / 255.0))




class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.opt = opter
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet(intInput,intfsize):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intInput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intInput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intfsize, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intfsize, out_channels=intfsize, kernel_size=3, stride=1, padding=1)
            )
        # end


        # ----------------- Part for adaptiva-kernel E-D --------------------------
            # ------------------- Encoder Part -----------------
        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            # ------------------- Decoder Part -----------------
        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )



        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet(64,51)
        self.moduleVertical2 = Subnet(64,51)
        self.moduleHorizontal1 = Subnet(64,51)
        self.moduleHorizontal2 = Subnet(64,51)

        self.modulePad_a = torch.nn.ReplicationPad2d(
            [int(math.floor(6)), int(math.floor(6)), int(math.floor(6)), int(math.floor(6))])
        self.modulePad_b = torch.nn.ReplicationPad2d(
            [int(math.floor(12)), int(math.floor(12)), int(math.floor(12)), int(math.floor(12))])
        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(25)), int(math.floor(25)), int(math.floor(25)), int(math.floor(25)) ])


        # ----------------- Part for optical flow prediction ------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, tensorInput1, tensorInput2):
        '''
        tensorInput1/2 : [bcz, 3, height, width]
        diff:            [bcz, 2, height, width]
        '''
        

        tensorJoin = torch.cat([ tensorInput1, tensorInput2 ], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)#[32, 128, 128]
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)#[64, 64, 64]
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)#[128, 32, 32]
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)#[256, 16, 16]
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)#[512, 8, 8]
        tensorPool5 = self.modulePool5(tensorConv5)

        

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)#[512, 4, 4]
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)#[512, 8, 8]

        tensorCombine = tensorUpsample5 + tensorConv5#[512, 8, 8]

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)#[256, 8, 8]
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)#[256, 16, 16]

        tensorCombine = tensorUpsample4 + tensorConv4#[256, 16, 16]

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)#[128, 16, 16]
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)#[128, 32, 32]

        tensorCombine = tensorUpsample3 + tensorConv3#[128, 32, 32]

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)#[64, 32, 32]
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)#[64, 64, 64]

        tensorCombine = tensorUpsample2 + tensorConv2#[64, 64, 64]

        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(tensorInput1), self.moduleVertical1(tensorCombine),
                                                self.moduleHorizontal1(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(tensorInput2), self.moduleVertical2(tensorCombine),
                                               self.moduleHorizontal2(tensorCombine))
        
        return tensorDot1 + tensorDot2

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    # ------------- Part for tensorboard --------------
    # writer = SummaryWriter(log_dir='tb/vimeo_baseline')
    # ------------- Part for tensorboard --------------
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(gpu)

    BATCH_SIZE=batch_size
    EPOCH=epoch

    LEARNING_RATE = lr
    belta1 = 0.9
    belta2 = 0.999

    trainset = vimeodataset(train_set, 'filelist.txt',transform_train)
    valset = vimeodataset(valid_set, 'test.txt')
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)


    SepConvNet = Network().cuda()
    # SepConvNet.apply(weights_init)
    SepConvNet.load_state_dict(torch.load('vimeo_baseline_iter26-ltype_fSATD_fs-lr_0.001-trainloss_0.0933-evalloss_0.0933-evalpsnr_inf.pkl'))

    # MSE_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-6)

    # ----------------  Time part -------------------
    start_time = time.time()
    global_step = 0
    # ----------------  Time part -------------------


    # ---------------- Opt part -----------------------
    # opter = Opter(gpu)
    # -------------------------------------------------

    for epoch in range(0,EPOCH):
        SepConvNet.train().cuda()
        cnt = 0
        sumloss = 0.0 # The sumloss is for the whole training_set
        tsumloss = 0.0 # The tsumloss is for the printinterval
        printinterval = 300
        print("---------------[Epoch%3d]---------------"%(epoch + 1))
        for label_list in trainLoader:
            bad_list = label_list[7:]
            label_list = label_list[:7]
            # IPython.embed()
            # exit()
            global_step = global_step + 1
            cnt = cnt + 1
            SepConvNet_optimizer.zero_grad()
            loss_s = []
            for i in range(5):
                imgL = var(bad_list[i]).cuda()
                imgR = var(bad_list[i+1]).cuda()
                label = var(label_list[i+2]).cuda()
                
                output = SepConvNet(imgL, imgR)
                loss_s.append(SepConvNet_cost(output, label))

            loss = (loss_s[0] + loss_s[1] + loss_s[2] + loss_s[3] + loss_s[4]) / 5
            loss.backward()
            SepConvNet_optimizer.step()
            
            sumloss = sumloss + loss.data.item()
            tsumloss = tsumloss + loss.data.item()
            # print("finish ", cnt)

            if cnt % printinterval == 0:
                # writer.add_image("Ref1 image", imgL[0], cnt)
                # writer.add_image("Ref2 image", imgR[0], cnt)
                # writer.add_image("Pred image", output[0], cnt)
                # writer.add_image("Target image", label[0], cnt)
                # writer.add_scalar('Train Batch SATD loss', loss.data.item(), int(global_step / printinterval))
                # writer.add_scalar('Train Interval SATD loss', tsumloss / printinterval, int(global_step / printinterval))
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
        for label_list in valLoader:
            bad_list = label_list[7:]
            label_list = label_list[:7]
            loss_s = []
            with torch.no_grad():
                for i in range(5):
                    imgL = var(bad_list[i]).cuda()
                    imgR = var(bad_list[i+1]).cuda()
                    label = var(label_list[i+2]).cuda()
                    output = SepConvNet(imgL, imgR)
                    psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                    loss_s.append(SepConvNet_cost(output, label))
                
                loss = (loss_s[0] + loss_s[1] + loss_s[2] + loss_s[3] + loss_s[4])
                sumloss = sumloss + loss.data.item()
                evalcnt = evalcnt + 5
        # ------------- Tensorboard part -------------
        # writer.add_scalar("Valid SATD loss", sumloss / evalcnt, epoch)
        # writer.add_scalar("Valid PSNR", psnr / valset.__len__(), epoch)
        # ------------- Tensorboard part -------------
        print('Validation loss [%.6f],  Average PSNR [%.4f]' % (
        sumloss / evalcnt, psnr / evalcnt))
        SepConvNet_schedule.step(psnr / evalcnt)
        torch.save(SepConvNet.state_dict(),
                os.path.join('.', 'bv_vimeo_baseline_iter' + str(epoch + 1)
                                + '-ltype_fSATD_fs'
                                + '-lr_' + str(LEARNING_RATE)
                                + '-trainloss_' + str(round(trainloss, 4))
                                + '-evalloss_' + str(round(sumloss / evalcnt, 4))
                                + '-evalpsnr_' + str(round(psnr / evalcnt, 4)) + '.pkl'))
    # writer.close()

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
    parser.add_argument("--train_set", type=str, dest="train_set", default="/data1/ikusyou/vimeo_7_train", 
                        help="Path of the training set")
    parser.add_argument("--valid_set", type=str, dest="valid_set", default="/data1/ikusyou/vimeo_7_valid", 
                        help="Path of the validation set")

    args = parser.parse_args()
    main(**vars(args))