'''
In this version, I will predict different flow for frame1/2
To compare the performance with unified flow by just multiply by factor 2.0
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
from torch.nn import BatchNorm2d
import IPython

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
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


def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        height = 128
        width = 128
        batch_size = 16

        x_t = torch.matmul(
            torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
        y_t = torch.matmul(
            torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

        self.grid_x = x_t.view(1, height, width)
        self.grid_y = y_t.view(1, height, width)
        self.grid_x = torch.autograd.Variable(
                self.grid_x.repeat([batch_size, 1, 1])).cuda()
        self.grid_y = torch.autograd.Variable(
                self.grid_y.repeat([batch_size, 1, 1])).cuda()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_bn = BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = BatchNorm2d(256)

        self.deconv1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv1_bn = BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(
            384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(
            192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv3_bn = BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 5, kernel_size=5, stride=1, padding=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, tensorInput1, tensorInput2, valid):
        x = torch.cat([tensorInput1, tensorInput2], 1)
        input = x
        input_size = tuple(x.size()[2:4])
        batch_size = tensorInput1.size(0)
        height = tensorInput1.size(2)
        width = tensorInput1.size(3)
        # IPython.embed()
        # exit()
        x = self.conv1(x)
    
        x = self.conv1_bn(x)
        conv1 = self.relu(x)

        x = self.pool(conv1)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)

        x = self.pool(conv2)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        conv3 = self.relu(x)

        x = self.pool(conv3)

        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = nn.functional.tanh(x)

        flow1 = x[:, 0:2, :, :]
        flow2 = x[:, 2:4, :, :]
        mask  = x[:, 4:5, :, :]

        flow1 = 0.5 * flow1
        flow2 = 0.5 * flow2

        
        flow_x1 = flow1[:,0] * (128.0 / float(width))
        flow_y1 = flow1[:,1] * (128.0 / float(height))

        flow_x2 = flow2[:,0] * (128.0 / float(width))
        flow_y2 = flow2[:,1] * (128.0 / float(height))

        if valid:
            grid_x, grid_y = meshgrid(input_size[0], input_size[1])
            grid_x = torch.autograd.Variable(
                grid_x.repeat([input.size()[0], 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
                grid_y.repeat([input.size()[0], 1, 1])).cuda()
            coor_x_1 = grid_x - flow_x1
            coor_y_1 = grid_y - flow_y1
            coor_x_2 = grid_x - flow_x2
            coor_y_2 = grid_y - flow_y2
        else:
            coor_x_1 = self.grid_x[:batch_size] - flow_x1
            coor_y_1 = self.grid_y[:batch_size] - flow_y1
            coor_x_2 = self.grid_x[:batch_size] - flow_x2
            coor_y_2 = self.grid_y[:batch_size] - flow_y2


        output_1 = torch.nn.functional.grid_sample(
            input[:, 0:3, :, :],
            torch.stack([coor_x_1, coor_y_1], dim=3),
            padding_mode='border')
        output_2 = torch.nn.functional.grid_sample(
            input[:, 3:6, :, :],
            torch.stack([coor_x_2, coor_y_2], dim=3),
            padding_mode='border')

        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])
        x = mask * output_1 + (1.0 - mask) * output_2

        return x



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data,0.0)

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    # ------------- Part for tensorboard --------------
    writer = SummaryWriter(log_dir="tb/dual_voxel_base")
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

    # MSE_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-6)

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

            output = SepConvNet(imgL, imgR, False)
            loss = SepConvNet_cost(output, label)
            loss.backward()
            SepConvNet_optimizer.step()

            sumloss = sumloss + loss.data.item()
            tsumloss = tsumloss + loss.data.item()
            
            if cnt % printinterval == 0:
                writer.add_image("Ref1 image", imgL[0], cnt)
                writer.add_image("Ref2 image", imgR[0], cnt)
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

                output = SepConvNet(imgL, imgR, True)
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
                os.path.join('.', 'dual_voxel_base_iter' + str(epoch + 1)
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
