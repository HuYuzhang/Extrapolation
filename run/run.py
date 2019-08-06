#!/usr/bin/env python
'''
Usage: python run.py im1Name im2Name outName
'''
import sys
sys.path.append("D:\\hyz\\19Summer\\ISCAS\\Extrapolation\\utils")
import getopt
import math
# import numpy
import shutil
import os
import torch
import time
from torch.autograd import Variable as var
import numpy as np
import torch.nn.functional as func
import torch.nn as nn
import cv2
from Opt import Opter
# import IPython
##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0
torch.cuda.set_device(0)
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

if __name__ == '__main__':
    arguments_strFirst = sys.argv[1]
    arguments_strSecond = sys.argv[2]
    arguments_strOut = sys.argv[3]
    print(sys.argv)
    # lqp = float(arguments_strFirst[:-4].split('_')[-1])/51.0
    # rqp = float(arguments_strSecond[:-4].split('_')[-1])/51.0
    img1 = cv2.imread(arguments_strFirst)[:,:,::-1]
    img2 = cv2.imread(arguments_strSecond)[:,:,::-1]
    
    # diff = np.zeros([img1.shape[0], img1.shape[1], 1])
    img1 = var(torch.from_numpy(img1.transpose(2,0,1).astype(np.float32) / 255.0))
    img1 = img1.view(1,img1.size(0),img1.size(1),img1.size(2))

    img2 = var(torch.from_numpy(img2.transpose(2,0,1).astype(np.float32) / 255.0))
    img2 = img2.view(1, img2.size(0), img2.size(1), img2.size(2))

    opt = Opter()

    flo = opt.estimate(img2, img1)
    warped = opt.warp(img2, flo).cpu().detach().numpy()[0].transpose(1,2,0) * 255.0
    warped = warped.astype('uint8')[:,:,::-1]

    cv2.imwrite(arguments_strOut, warped)
