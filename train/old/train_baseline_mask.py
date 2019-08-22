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


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

class Opter():
    def __init__(self, gpu=0):
        torch.cuda.set_device(gpu)
        self.Backward_tensorGrid = {}
        def Backward(tensorInput, tensorFlow):
            if str(tensorFlow.size()) not in self.Backward_tensorGrid:
                tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                self.Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
            # end

            tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

            return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        # end

        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()

                class Preprocess(torch.nn.Module):
                    def __init__(self):
                        super(Preprocess, self).__init__()
                    # end

                    def forward(self, tensorInput):
                        tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                        tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                        tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

                        return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
                    # end
                # end

                class Basic(torch.nn.Module):
                    def __init__(self, intLevel):
                        super(Basic, self).__init__()

                        self.moduleBasic = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                        )
                    # end

                    def forward(self, tensorInput):
                        return self.moduleBasic(tensorInput)
                    # end
                # end

                self.modulePreprocess = Preprocess()

                self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
                try:
                    self.load_state_dict(torch.load('network-sintel-final.pytorch'))
                except Exception:
                    self.load_state_dict(torch.load('network-sintel-final.pytorch'))
            # end

            def forward(self, tensorFirst, tensorSecond):
                tensorFlow = []

                tensorFirst = [ self.modulePreprocess(tensorFirst) ]
                tensorSecond = [ self.modulePreprocess(tensorSecond) ]

                for intLevel in range(5):
                    if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                        tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                        tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2, count_include_pad=False))
                    # end
                # end

                tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

                for intLevel in range(len(tensorFirst)):
                    tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

                    if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
                    if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

                    tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
                # end

                return tensorFlow
            # end
        # end
        self.FpyNet = Network().cuda().eval()
    
    def estimate(self, tensorFirst, tensorSecond):
        '''
        The input can be Tensor of size: [bcz, c, h, w] or [c, h, w], and I will transfer them to .cuda()
        '''
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        if len(tensorFirst.size()) == 3:
            tensorFirst = tensorFirst.unsqueeze(0)
            tensorSecond = tensorSecond.unsqueeze(0)
        
        intWidth = tensorFirst.size(3)
        intHeight = tensorFirst.size(2)
        batch_size = tensorFirst.size(0)
        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tensorPreprocessedFirst = tensorFirst.view(batch_size, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(batch_size, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tensorFlow = torch.nn.functional.interpolate(input=self.FpyNet(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, tensorInput, tensorFlow):
        '''
        The input can be Tensor of size: [bcz, c, h, w] or [c, h, w], and I will transfer them to .cuda()
        '''
        tensorInput = tensorInput.cuda()
        tensorFlow = tensorFlow.cuda()
        if len(tensorInput.size()) == 3:
            tensorInput = tensorInput.unsqueeze(0)
            tensorFlow = tensorFlow.unsqueeze(0)

        if str(tensorFlow.size()) not in self.Backward_tensorGrid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        # end
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


        # ----------------- Part for optical flow E-D ------------------------
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1_bn = BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2_bn = BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3_bn = BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bottleneck_bn = BatchNorm2d(256)

        self.deconv1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.deconv1_bn = BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(
            384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        # self.deconv2_bn = BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(
            192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.deconv3_bn = BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        # Above is the predicted optical flow
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
        x = tensorJoin
        x = self.conv1(x)
    
        # x = self.conv1_bn(x)
        conv1 = self.relu(x)

        x = self.pool(conv1)

        x = self.conv2(x)
        # x = self.conv2_bn(x)
        conv2 = self.relu(x)

        x = self.pool(conv2)

        x = self.conv3(x)
        # x = self.conv3_bn(x)
        conv3 = self.relu(x)

        x = self.pool(conv3)

        x = self.bottleneck(x)
        # x = self.bottleneck_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        # x = self.deconv1_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        # x = self.deconv2_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        # x = self.deconv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x)
        mask = nn.functional.tanh(x)
        # ---------------- Predict the back-forward optical flow and warp the inputFrame2                         Part1
        
        # ---------------- Predict the back-forward optical flow and warp the inputFrame2                         Part1



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
        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])
        x = mask * tensorDot1 + (1.0 - mask) * tensorDot2

        return x



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data,0.0)

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    # ------------- Part for tensorboard --------------
    writer = SummaryWriter(log_dir='tb/ft1_baseline_mask')
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
    # SepConvNet.apply(weights_init)
    SepConvNet.load_state_dict(torch.load('/mnt/hdd/iku/ISCAS/train/mask_baseline_iter52-ltype_fSATD_fs-lr_0.001-trainloss_0.1279-evalloss_0.1181-evalpsnr_29.6526.pkl'))

    # MSE_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-7)

    # ----------------  Time part -------------------
    start_time = time.time()
    global_step = 0
    # ----------------  Time part -------------------


    # ---------------- Opt part -----------------------
    opter = Opter(gpu)
    # -------------------------------------------------

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
            
            output = SepConvNet(imgL, imgR)
            loss = SepConvNet_cost(output, label)
            loss.backward()
            SepConvNet_optimizer.step()
            
            sumloss = sumloss + loss.data.item()
            tsumloss = tsumloss + loss.data.item()


            if cnt % printinterval == 0:
                writer.add_image("Ref image", imgR[0], cnt)
                writer.add_image("Pred image", output[0], cnt)
                writer.add_image("Target image", label[0], cnt)
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

                output = SepConvNet(imgL, imgR)
                loss = SepConvNet_cost(output, label)

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
                os.path.join('.', 'ft1_mask_baseline_iter' + str(epoch + 1)
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