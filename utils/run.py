#!/usr/bin/env python

import getopt
import math
import numpy
import shutil
import os
import PIL
import PIL.Image
import sys
import torch
import time
from torch.autograd import Variable as var
import numpy as np
import sepconv
import torch.nn.functional as func
import torch.nn as nn


##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

# torch.cuda.device(0) # change this if you have a multiple graphics cards and you want to utilize them

# torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = '37'
arguments_strFirst = './l_5_8_29.bmp'
arguments_strSecond = './r_5_8_29.bmp'
arguments_strOut = './result.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model':
        arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details

    elif strOption == '--first':
        arguments_strFirst = strArgument # path to the first frame

    elif strOption == '--second':
        arguments_strSecond = strArgument # path to the second frame

    elif strOption == '--out':
        arguments_strOut = strArgument # path to where the output should be stored

    # end
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

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

        def Subnet(intInput, intfsize):
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

        self.mv1_a = Subnet(256, 13)
        self.mv2_a = Subnet(256, 13)
        self.mh1_a = Subnet(256, 13)
        self.mh2_a = Subnet(256, 13)
        self.w_a1 = Subnet(258, 2)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.mv1_b = Subnet(128, 25)
        self.mv2_b = Subnet(128, 25)
        self.mh1_b = Subnet(128, 25)
        self.mh2_b = Subnet(128, 25)
        self.w_b1 = Subnet(130, 2)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet(64, 51)
        self.moduleVertical2 = Subnet(64, 51)
        self.moduleHorizontal1 = Subnet(64, 51)
        self.moduleHorizontal2 = Subnet(64, 51)
        self.w1 = Subnet(66, 2)

        self.modulePad_a = torch.nn.ReplicationPad2d(
            [int(math.floor(6)), int(math.floor(6)), int(math.floor(6)), int(math.floor(6))])
        self.modulePad_b = torch.nn.ReplicationPad2d(
            [int(math.floor(12)), int(math.floor(12)), int(math.floor(12)), int(math.floor(12))])
        self.modulePad = torch.nn.ReplicationPad2d(
            [int(math.floor(25)), int(math.floor(25)), int(math.floor(25)), int(math.floor(25))])

    def forward(self, tensorInput1, tensorInput2, img1map, img2map):
        tensorJoin = torch.cat([tensorInput1, tensorInput2], 1)
        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDot1_a = sepconv.FunctionSepconv_cpu()(self.modulePad_a(
            func.upsample(tensorInput1, size=(tensorInput1.shape[2] // 4, tensorInput1.shape[3] // 4), mode='bilinear',
                          align_corners=True)),
            self.mv1_a(tensorCombine), self.mh1_a(tensorCombine))
        tensorDot2_a = sepconv.FunctionSepconv_cpu()(self.modulePad_a(
            func.upsample(tensorInput2, size=(tensorInput1.shape[2] // 4, tensorInput1.shape[3] // 4), mode='bilinear',
                          align_corners=True)), self.mv2_a(tensorCombine), self.mh2_a(tensorCombine))
        w_a1 = self.w_a1(torch.cat([tensorCombine, func.upsample(img1map, size=(tensorInput1.shape[2] // 8, tensorInput1.shape[3] // 8), mode='bilinear',
                          align_corners=True), func.upsample(img2map, size=(tensorInput1.shape[2] // 8, tensorInput1.shape[3] // 8), mode='bilinear',
                          align_corners=True)], 1))
        res_a1 = tensorDot1_a.mul(w_a1[:,0,:,:].unsqueeze(1)) + \
               tensorDot2_a.mul(w_a1[:,1,:,:].unsqueeze(1))


        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDot1_b = sepconv.FunctionSepconv_cpu()(self.modulePad_b(
            func.upsample(tensorInput1, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)), self.mv1_b(tensorCombine), self.mh1_b(tensorCombine))+func.upsample(res_a1, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)
        tensorDot2_b = sepconv.FunctionSepconv_cpu()(self.modulePad_b(
            func.upsample(tensorInput2, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)), self.mv2_b(tensorCombine), self.mh2_b(tensorCombine))+func.upsample(res_a1, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)
        w_b1 = self.w_b1(torch.cat([tensorCombine, func.upsample(img1map, size=(tensorInput1.shape[2] // 4, tensorInput1.shape[3] // 4), mode='bilinear',
                          align_corners=True),func.upsample(img2map, size=(tensorInput1.shape[2] // 4, tensorInput1.shape[3] // 4), mode='bilinear',
                          align_corners=True)], 1))
        res_b1 = tensorDot1_b.mul(w_b1[:,0,:,:].unsqueeze(1)) + tensorDot2_b.mul(w_b1[:,1,:,:].unsqueeze(1))

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        tensorDot1 = sepconv.FunctionSepconv_cpu()(self.modulePad(tensorInput1), self.moduleVertical1(tensorCombine),
                                               self.moduleHorizontal1(tensorCombine))+func.upsample(res_b1, size=(tensorInput1.shape[2], tensorInput1.shape[3]), mode='bilinear',
                          align_corners=True)
        tensorDot2 = sepconv.FunctionSepconv_cpu()(self.modulePad(tensorInput2), self.moduleVertical2(tensorCombine),
                                               self.moduleHorizontal2(tensorCombine))+func.upsample(res_b1, size=(tensorInput1.shape[2], tensorInput1.shape[3]), mode='bilinear',
                          align_corners=True)
        w1 = self.w1(torch.cat([tensorCombine, func.upsample(img1map, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True),func.upsample(img2map, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)], 1))
        res1 = tensorDot1.mul(w1[:,0,:,:].unsqueeze(1)) + tensorDot2.mul(w1[:,1,:,:].unsqueeze(1))

        return res1

SepConvNet = Network().eval()
SepConvNet.load_state_dict(torch.load('./SepConv_iter70-lSATD_51_QA_MASCNN_Minter_LD-lr_0.0001-trainloss_0.136503-evalloss_0.126263-evalpsnr_30.4174.pkl', map_location='cpu'))

if __name__ == '__main__':

    lqp = float(arguments_strFirst[:-4].split('_')[-1])/51.0
    rqp = float(arguments_strSecond[:-4].split('_')[-1])/51.0
    img1 = np.array(PIL.Image.open(arguments_strFirst))
    img1 = var(torch.from_numpy(img1.transpose(2,0,1).astype(np.float32) / 255.0))
    img1 = img1.view(1,img1.size(0),img1.size(1),img1.size(2))
    img2 = np.array(PIL.Image.open(arguments_strSecond))
    img2 = var(torch.from_numpy(img2.transpose(2,0,1).astype(np.float32) / 255.0))
    img2 = img2.view(1, img2.size(0), img2.size(1), img2.size(2))

    assert (img1.size(1)==img2.size(1))
    assert (img1.size(2) == img2.size(2))
    assert (img1.size(3) == img2.size(3))

    intchannel = img1.size(1)
    inthei = img1.size(2)
    intwid = img1.size(3)

    intpaddingwidth = 0
    intpaddingheight = 0

    if inthei != (inthei >> 5) << 5:
        intpaddingheight = (((inthei >> 5) + 1) << 5) - inthei
    if intwid != (intwid >> 5) << 5:
        intpaddingwidth = (((intwid >> 5) + 1) << 5) - intwid

    intpaddingleft = int(intpaddingwidth/2)
    intpaddingright = int(intpaddingwidth/2)
    intpaddingtop = int(intpaddingheight/2)
    intpaddingbottom = int(intpaddingheight/2)


    modulePaddingInput = torch.nn.Sequential()
    modulePaddingOutput = torch.nn.Sequential()

    modulePaddingInput = torch.nn.ReplicationPad2d(padding=[intpaddingleft, intpaddingright, intpaddingtop, intpaddingbottom])
    modulePaddingOutput = torch.nn.ReplicationPad2d(padding=[0 - intpaddingleft, 0 - intpaddingright, 0 - intpaddingtop,0 - intpaddingbottom])

    img1 = modulePaddingInput(img1)
    img2 = modulePaddingInput(img2)
    img1map = torch.zeros(1, 1, img1.size(2), img1.size(3)) + lqp
    img2map = torch.zeros(1, 1, img1.size(2), img1.size(3)) + rqp

    output = SepConvNet(img1, img2, img1map, img2map)
    output = modulePaddingOutput(output)
    output = output.squeeze(0)
    PIL.Image.fromarray((output.clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)).save(arguments_strOut)
