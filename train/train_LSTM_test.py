'''
Run on X5


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
            cropy = random.randint(2, 60)
            cropx = random.randint(2, 180)

            for i in range(7):
                raw_imgs[i] = raw_imgs[i][cropy:cropy + 128, cropx:cropx + 128, :]
                drop_imgs[i] = drop_imgs[i][cropy:cropy + 128, cropx:cropx + 128, :]

            # IPython.embed()
            # exit()
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

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
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

    def __init__(self, input_dim=3, hidden_dim=32, kernel_size=(3,3)):
        super(ConvLSTM, self).__init__()


        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.cell = ConvLSTMCell(   input_dim=self.input_dim,
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

        self.mv1_a_ = Subnet(384,13)
        self.mv2_a_ = Subnet(384, 13)
        self.mh1_a_ = Subnet(384, 13)
        self.mh2_a_ = Subnet(384, 13)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.mv1_b_ = Subnet(192, 25)
        self.mv2_b_ = Subnet(192, 25)
        self.mh1_b_ = Subnet(192, 25)
        self.mh2_b_ = Subnet(192, 25)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1_ = Subnet(96,51)
        self.moduleVertical2_ = Subnet(96,51)
        self.moduleHorizontal1_ = Subnet(96,51)
        self.moduleHorizontal2_ = Subnet(96,51)

        self.modulePad_a = torch.nn.ReplicationPad2d(
            [int(math.floor(6)), int(math.floor(6)), int(math.floor(6)), int(math.floor(6))])
        self.modulePad_b = torch.nn.ReplicationPad2d(
            [int(math.floor(12)), int(math.floor(12)), int(math.floor(12)), int(math.floor(12))])
        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(25)), int(math.floor(25)), int(math.floor(25)), int(math.floor(25)) ])
        # ------------------- LSTM Part -----------------
        # Note that all new layers should be put here to keep the instant of the baseline's layer~
        self.moduleLSTM = ConvLSTM(3, 32)
        self.moduleConvH0 = Basic(32, 32)
        self.moduleDownH0 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConvH1 = Basic(32, 64)
        self.moduleDownH1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConvH2 = Basic(64, 128)
        self.moduleDownH2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        # ------------------- Initialize Part -----------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        # LSTM part 

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

        # ------------------------- I use the convolution with stride of 2 to work as a downsample function~, which accords to the resolution of [128, 128], [64, 64], [32, 32]
        tensorL0 = self.moduleDownH0(self.moduleConvH0(tensorH_next))
        tensorL1 = self.moduleDownH1(self.moduleConvH1(tensorL0))
        tensorL2 = self.moduleDownH2(self.moduleConvH2(tensorL1))
        # tensorL1 = self.moduleDownLSTM1(tensorL0)
        # tensorL2 = self.moduleDownLSTM2(tensorL1)
        # ------------------------- I use the convolution with stride of 2 to work as a downsample function~, which accords to the resolution of [128, 128], [64, 64], [32, 32]


        tensorJoin = torch.cat([ tensorInput1, tensorInput2 ], 1)
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
        
        # ------- LSTM combine ------------
        tensorCombineL2 = torch.cat([tensorCombine, tensorL2], 1) # This channel is 256 + 128 = 384
        # ------- LSTM combine ------------

        tensorDot1_a = sepconv.FunctionSepconv()(self.modulePad_a(func.upsample(tensorInput1,size=(tensorInput1.shape[2]//4,tensorInput1.shape[3]//4),mode='bilinear',align_corners=True)),
                                                  self.mv1_a_(tensorCombineL2),self.mh1_a_(tensorCombineL2))
        tensorDot2_a = sepconv.FunctionSepconv()(self.modulePad_a(func.upsample(tensorInput2, size=(tensorInput1.shape[2] // 4, tensorInput1.shape[3] // 4), mode='bilinear',
                          align_corners=True)), self.mv2_a_(tensorCombineL2), self.mh2_a_(tensorCombineL2))

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        # ------- LSTM combine ------------
        tensorCombineL1 = torch.cat([tensorCombine, tensorL1], 1) # This channel is 128 + 64 = 192
        # ------- LSTM combine ------------
        tensorDot1_b = sepconv.FunctionSepconv()(self.modulePad_b(func.upsample(tensorInput1, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                           align_corners=True)),self.mv1_b_(tensorCombineL1), self.mh1_b_(tensorCombineL1))
        tensorDot2_b = sepconv.FunctionSepconv()(self.modulePad_b(func.upsample(tensorInput2, size=(tensorInput1.shape[2] // 2, tensorInput1.shape[3] // 2), mode='bilinear',
                          align_corners=True)), self.mv2_b_(tensorCombineL1), self.mh2_b_(tensorCombineL1))

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2
        # ------- LSTM combine ------------
        tensorCombineL0 = torch.cat([tensorCombine, tensorL0], 1) # This channel is 64 + 32 = 96
        # ------- LSTM combine ------------
        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(tensorInput1), self.moduleVertical1_(tensorCombineL0),
                                                self.moduleHorizontal1_(tensorCombineL0))
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(tensorInput2), self.moduleVertical2_(tensorCombineL0),
                                                self.moduleHorizontal2_(tensorCombineL0))

        return tensorDot1 + tensorDot2, tensorDot1_a + tensorDot2_a, tensorDot1_b + tensorDot2_b, (tensorH_next, tensorC_next)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # print('load: ', name)
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)


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
    valLoader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE*2, shuffle=False)
    assert(len(valset) % BATCH_SIZE == 0)


    SepConvNet = Network().cuda()
    # SepConvNet.apply(weights_init)
    # SepConvNet.load_my_state_dict(torch.load('SepConv_iter95-ltype_fSATD_fs-lr_0.0001-trainloss_0.1441-evalloss_0.1324-evalpsnr_29.9585.pkl', map_location="cuda:%d"%(gpu)))
    SepConvNet.load_my_state_dict(torch.load('SepConv_iter95-ltype_fSATD_fs-lr_0.0001-trainloss_0.1441-evalloss_0.1324-evalpsnr_29.9585.pkl', map_location="cuda:%d"%(gpu)))

    # MSE_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    

    child_cnt = 0

    skip_childs = list(set(range(33)) - set([14,15,16,17,  20,21,22,23,  26,27,28,29]))
    for child in SepConvNet.children():
        # print('-----------  Children:%d ----------------'%(child_cnt))
        # print(child)
        param_cnt = 0
        if not child_cnt in skip_childs:
            for param in child.parameters():
                # print("Param: %d in child: %d is frozen"%(param_cnt, child_cnt))
                param.requires_grad = False
                param_cnt += 1
        child_cnt += 1
        
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    # SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_optimizer = optim.Adamax(filter(lambda p: p.requires_grad, SepConvNet.parameters()),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-6)
    # IPython.embed()
    # exit()
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
            loss_s = []
            for i in range(5):
                imgL = var(bad_list[i]).cuda()
                imgR = var(bad_list[i+1]).cuda()
                label = var(label_list[i+2]).cuda()
                poor_label = var(bad_list[i+2]).cuda()
                if i == 0:
                    SepConvNet_optimizer.zero_grad()

                    output, output_a, output_b, stat = SepConvNet(imgL, imgR)
                    res = poor_label - output
                    # loss = SepConvNet_cost(output, label)
                    loss = 0.5*SepConvNet_cost(output, label) + \
                            0.2*SepConvNet_cost(output_a,func.upsample(label, size=(label.shape[2] // 4, label.shape[3] // 4), mode='bilinear',align_corners=True)) + \
                            0.3*SepConvNet_cost(output_b,func.upsample(label, size=(label.shape[2] // 2, label.shape[3] // 2), mode='bilinear',align_corners=True))

                    loss.backward(retain_graph=True)
                    SepConvNet_optimizer.step()

                    sumloss = sumloss + loss.data.item()
                    tsumloss = tsumloss + loss.data.item()

                elif i < 4:
                    SepConvNet_optimizer.zero_grad()

                    output, output_a, output_b, stat = SepConvNet(imgL, imgR, res, stat)
                    res = poor_label - output
                    # loss = SepConvNet_cost(output, label)
                    loss = 0.5*SepConvNet_cost(output, label) + \
                            0.2*SepConvNet_cost(output_a,func.upsample(label, size=(label.shape[2] // 4, label.shape[3] // 4), mode='bilinear',align_corners=True)) + \
                            0.3*SepConvNet_cost(output_b,func.upsample(label, size=(label.shape[2] // 2, label.shape[3] // 2), mode='bilinear',align_corners=True))
                            
                    loss.backward(retain_graph=True)
                    SepConvNet_optimizer.step()

                    sumloss = sumloss + loss.data.item()
                    tsumloss = tsumloss + loss.data.item()
                else:
                    SepConvNet_optimizer.zero_grad()

                    output, output_a, output_b, stat = SepConvNet(imgL, imgR, res, stat)
                    res = poor_label - output
                    # loss = SepConvNet_cost(output, label)
                    loss = 0.5*SepConvNet_cost(output, label) + \
                            0.2*SepConvNet_cost(output_a,func.upsample(label, size=(label.shape[2] // 4, label.shape[3] // 4), mode='bilinear',align_corners=True)) + \
                            0.3*SepConvNet_cost(output_b,func.upsample(label, size=(label.shape[2] // 2, label.shape[3] // 2), mode='bilinear',align_corners=True))

                    loss.backward()
                    SepConvNet_optimizer.step()

                    sumloss = sumloss + loss.data.item()
                    tsumloss = tsumloss + loss.data.item()
            
            
            
            if cnt % printinterval == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Time [%4.4f], Batch loss [%.6f], Interval loss [%.6f]' %
                    (epoch + 1, EPOCH, cnt, len(trainset) // BATCH_SIZE, time.time() - start_time, loss.data.item(), tsumloss / printinterval / 5))
                tsumloss = 0.0
        print('Epoch [%d/%d], iter: %d, Time [%4.4f], Avg Loss [%.6f]' %
            (epoch + 1, EPOCH, cnt, time.time() - start_time, sumloss / cnt / 5))


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
                        output, output_a, output_b, stat = SepConvNet(imgL, imgR)
                        psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                        res = poor_label - output

                        loss = SepConvNet_cost(output, label)
                        sumloss = sumloss + loss.data.item()

                    else:
                        output, output_a, output_b, stat = SepConvNet(imgL, imgR, res, stat)
                        psnr = psnr + calcPSNR.calcPSNR(output.cpu().data.numpy(), label.cpu().data.numpy())
                        res = poor_label - output

                        loss = SepConvNet_cost(output, label)
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
                os.path.join('.', 'multiscale_test_LSTM_iter' + str(epoch + 1)
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
                        default=8, help="Mini-batch size")
    parser.add_argument("--epoch", type=int, dest="epoch",
                        default=200, help="Number of epochs")
    parser.add_argument("--gpu", type=int, dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--train_set", type=str, dest="train_set", default="/mnt/ssd/iku/vimeo_7_train_split_ssd", 
                        help="Path of the training set")
    parser.add_argument("--valid_set", type=str, dest="valid_set", default="/mnt/hdd/iku/vimeo_7_valid", 
                        help="Path of the validation set")

    args = parser.parse_args()
    main(**vars(args))

