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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        
        By HuYuzhang: Note that I don't change anything for the definition of this class: ConvLSTMCell
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        '''
        input_tensor：Normal data, like image etc
        cur_state: a tuple consist of (tensorH, tensorC)
        Note by Hu Yuzhang
        '''
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (var(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                var(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())


class ConvLSTM(nn.Module):
    '''
    By HuYuzhang: 
    Here I do a lot change...
    Now this class no more support multi-layer cell and multi-time-step input...
    Sounds like it becomes worse, but this change can benefit my work with warping or separate convolution etc
    Params:
    (tuple)input_size=(height, width)
    (int)input_dim=input_channels
    (int)hidden_dim=hidden_channels(Note the for the original version, this is a list for the support of the multi-layer)
    (tuple)kernel_size = (k_w, k_h)
    ! The removed params compared to the original version:
    num_layers=3, I don't need multi-layer, just like the change on the param: hidden_dim
    batch_first=3, I promise that all my input is of shape [bzs, c, h, w], and without time step, I will iter by myself
    bias=True, No doubt that I will use the bias...
    return_all_layers=False, Now that I only have one input, there is no conception of time... only one val will be returned
    '''         

    def __init__(self, input_size=(128,128), input_dim=3, hidden_dim=32, kernel_size=(3,3)):
        super(ConvLSTM, self).__init__()

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.cell = ConvLSTMCell(input_size=(self.height, self.width),
                                    input_dim=self.input_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size)


    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor:  
            4-D Tensor of shape (bcz, c, h, w)
        hidden_state: todo
            4-D Tensor of shape (bcz, hidden_dim, h, w) (For I use the default stride of 1)
            
        Returns
        -------
        layer_output, last_state
        """

        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3))


        h, c = self.cell(input_tensor=input_tensor, cur_state=hidden_state)
        # Note that h and c is the return value, where h is the output of LSTM and c is the new status of the cell
        return h, c


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

    
    # ------------------- Encoder Part -----------------
        self.moduleConv1 = Basic(38, 32) #
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

    # ------------------- LSTM Part -----------------
        self.moduleLSTM = ConvLSTM()

    # ------------------- Initialize Part -----------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, tensorInput1, tensorInput2, tensorResidual=None, tensorHidden=None):
        '''
        tensorInput1/2 : [bcz, 3, height, width]
        tensorResidual:  [bcz, 3, height, width]
        tensorHidden:(tuple or None) ([bcz, hidden_dim, height, width])
        When the LSTM_state is Noe, it means that its the first time step
        '''
        batch_size = tensorInput1.size(0)
    # ------------------- LSTM Part --------------------
        if tensorResidual is None:
            tensorResidual = var(torch.zeros(batch_size, tensorInput1.size(1), tensorInput1.size(2), tensorInput1.size(3))).cuda()
            tensorH_next, tensorC_next = self.moduleLSTM(tensorResidual) # Hence we also don't have the tensorHidden
        else:
            tensorH_next, tensorC_next = self.moduleLSTM(tensorResidual, tensorHidden)
        
    # ------------------- Encoder Part -----------------
        tensorJoin = torch.cat([ tensorInput1, tensorInput2, tensorH_next ], 1)

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

    # ------------------- Doceder Part -----------------
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
        
        
    # Return the predictd tensor and the next state of convLSTM
        return tensorDot1 + tensorDot2, (tensorH_next, tensorC_next)

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    # ------------- Part for tensorboard --------------
    # writer = SummaryWriter(log_dir='tb/LSTM_ft1')
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
    SepConvNet.load_state_dict(torch.load('ft1_LSTM_baseline_iter10-ltype_fSATD_fs-lr_0.001-trainloss_0.0871-evalloss_0.0868-evalpsnr_inf.pkl'))

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
                poor_label = var(bad_list[i+2]).cuda()
                if i == 0:
                    output, stat = SepConvNet(imgL, imgR)
                    res = poor_label - output
                    loss_s.append(SepConvNet_cost(output, label))
                else:
                    output, stat = SepConvNet(imgL, imgR, res, stat)
                    res = poor_label - output
                    loss_s.append(SepConvNet_cost(output, label))
            # IPython.embed()
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
                    poor_label = var(bad_list[i+2]).cuda()
                    if i == 0:
                        output, stat = SepConvNet(imgL, imgR)
                        psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                        res = poor_label - output
                        loss_s.append(SepConvNet_cost(output, label))
                    else:
                        output, stat = SepConvNet(imgL, imgR, res, stat)
                        psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                        res = poor_label - output
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
                os.path.join('.', 'bv_LSTM_poor_iter' + str(epoch + 1)
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