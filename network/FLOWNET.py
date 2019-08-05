'''
DENET's name come from "dual encoder network, which means use different encoder for optical flow and adaptive kernel
'''

import sys
sys.path.append('../utils')
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
import IPython

class Network(torch.nn.Module):
    def __init__(self, opter):
        super(Network, self).__init__()
        self.opt = opter
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



        # ----------------- Part for optical flow E-D ------------------------
        self.optConv1 = Basic(8, 32)
        self.optPool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.optConv2 = Basic(32, 64)
        self.optPool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.optConv3 = Basic(64, 128)
        self.optPool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.optConv4 = Basic(128, 256)
        self.optPool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.optConv5 = Basic(256, 512)
        self.optPool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

            # ------------------- Decoder Part -----------------
        self.optDeconv5 = Basic(512, 512)
        self.optUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.optDeconv4 = Basic(512, 256)
        self.optUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.optDeconv3 = Basic(256, 128)
        self.optUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.optDeconv2 = Basic(128, 64)
        self.optUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.optDeconv1 = Basic(64, 32)
        self.optUpsample1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.optPred = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
        )
        # Above is the predicted optical flow
        # ----------------- Part for optical flow prediction ------------------------


    def forward(self, diff, tensorInput1, tensorInput2):
        '''
        tensorInput1/2 : [bcz, 3, height, width]
        diff:            [bcz, 2, height, width]
        '''
        #diff *= 2.0 # @ I multiply it by 2 just for favor of warping

        tensorJoin = torch.cat([ tensorInput1, tensorInput2, diff ], 1)

        # ---------------- Predict the back-forward optical flow and warp the inputFrame2                         Part1
        tensorOptConv1 = self.optConv1(tensorJoin)#[32, 128, 128]
        tensorOptPool1 = self.optPool1(tensorOptConv1)

        tensorOptConv2 = self.optConv2(tensorOptPool1)#[64, 64, 64]
        tensorOptPool2 = self.optPool2(tensorOptConv2)

        tensorOptConv3 = self.optConv3(tensorOptPool2)#[128, 32, 32]
        tensorOptPool3 = self.optPool3(tensorOptConv3)

        tensorOptConv4 = self.optConv4(tensorOptPool3)#[256, 16, 16]
        tensorOptPool4 = self.optPool4(tensorOptConv4)

        tensorOptConv5 = self.optConv5(tensorOptPool4)#[512, 8, 8]
        tensorOptPool5 = self.optPool5(tensorOptConv5)


        tensorOptDeconv5 = self.optDeconv5(tensorOptPool5)
        tensorOptUpsample5 = self.optUpsample5(tensorOptDeconv5)
        tensorCombine = tensorOptUpsample5 + tensorOptConv5

        tensorOptDeconv4 = self.optDeconv4(tensorCombine)
        tensorOptUpsample4 = self.optUpsample4(tensorOptDeconv4)
        tensorCombine = tensorOptUpsample4 + tensorOptConv4

        tensorOptDeconv3 = self.optDeconv3(tensorCombine)
        tensorOptUpsample3 = self.optUpsample3(tensorOptDeconv3)
        tensorCombine = tensorOptUpsample3 + tensorOptConv3

        tensorOptDeconv2 = self.optDeconv2(tensorCombine)
        tensorOptUpsample2 = self.optUpsample2(tensorOptDeconv2)
        tensorCombine = tensorOptUpsample2 + tensorOptConv2

        tensorOptDeconv1 = self.optDeconv1(tensorCombine)
        tensorOptUpsample1 = self.optUpsample1(tensorOptDeconv1)
        tensorCombine = tensorOptUpsample1 + tensorOptConv1

        tensorOptPred1 = self.optPred(tensorCombine)

        # Warp the raw image
        tensorWarp1 = self.opt.warp(tensorOptPred1, tensorInput2)
        # ---------------- Predict the back-forward optical flow and warp the inputFrame2                         Part1



        return tensorOptPred1, tensorWarp1

