'''
Run on X5

Note that I froze the encoder-decoder's weights

Forward and backward share one LSTM module, so for the backward prediction, in fact no convLSTM calculation need to be done for the backward prediction.

Due to the limitation of the GPU, the batch_size here is set to 8
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
        self.moduleConv1 = Basic(6, 32) #
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

        self.moduleVertical11 = Subnet(96,51) # child 18
        self.moduleVertical22 = Subnet(96,51) # child 19 
        self.moduleHorizontal11 = Subnet(96,51) # child 20 
        self.moduleHorizontal22 = Subnet(96,51) # child 21 

        self.modulePad_a = torch.nn.ReplicationPad2d(
            [int(math.floor(6)), int(math.floor(6)), int(math.floor(6)), int(math.floor(6))])
        self.modulePad_b = torch.nn.ReplicationPad2d(
            [int(math.floor(12)), int(math.floor(12)), int(math.floor(12)), int(math.floor(12))])
        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(25)), int(math.floor(25)), int(math.floor(25)), int(math.floor(25)) ])

    # ------------------- LSTM Part -----------------
        self.moduleLSTM_fat = ConvLSTM(6, 32) # child 25 
        self.moduleConvH = Basic(32, 32)  # child 26 
        self.moduleDownH = torch.nn.AvgPool2d(kernel_size=2, stride=2) # child 27

        # self.moduleLSTM_b = ConvLSTM(3, 32) # child 28
        # self.moduleConvH_b = Basic(32, 32) # child 29
        # self.moduleDownH_b = torch.nn.AvgPool2d(kernel_size=2, stride=2) # child 30
    # ------------------- Initialize Part -----------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, tensorInput1, tensorInput2, direction, tensorResidual=None, tensorHidden=None):
        '''
        tensorInput1/2 : [bcz, 3, height, width]
        tensorResidual:  [bcz, 3, height, width]
        tensorHidden:(tuple or None) ([bcz, hidden_dim, height, width])
        When the LSTM_state is Noe, it means that its the first time step
        '''
        batch_size = tensorInput1.size(0)
    # ------------------- LSTM Part --------------------
        if tensorResidual is None:
            tensorResidual = var(torch.zeros(batch_size, tensorInput1.size(1) * 2, tensorInput1.size(2), tensorInput1.size(3))).cuda()
            if direction == 0:
                tensorH_next, tensorC_next = self.moduleLSTM_fat(tensorResidual)
            else:
                (tensorH_next, tensorC_next) = tensorHidden
        else:
            if direction == 0:
                tensorH_next, tensorC_next = self.moduleLSTM_fat(tensorResidual, tensorHidden)
            else:
                (tensorH_next, tensorC_next) = tensorHidden

        tensorCombine2 =   self.moduleDownH(self.moduleConvH(tensorH_next))
    # ------------------- Encoder Part -----------------
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

        tensorCombine1 = tensorUpsample2 + tensorConv2#[64, 64, 64]

        tensorCombine = torch.cat([tensorCombine1, tensorCombine2], 1)

        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(tensorInput1), self.moduleVertical11(tensorCombine),
                                                self.moduleHorizontal11(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(tensorInput2), self.moduleVertical22(tensorCombine),
                                                self.moduleHorizontal22(tensorCombine))
        
        
    # Return the predictd tensor and the next state of convLSTM
        return tensorDot1 + tensorDot2, (tensorH_next, tensorC_next)


    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
            # print('load: ', name)


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
    valLoader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    assert(len(valset) % BATCH_SIZE == 0)


    SepConvNet = Network().cuda()
    # SepConvNet.apply(weights_init)
    SepConvNet.load_my_state_dict(torch.load('tail_LSTM_iter15-ltype_fSATD_fs-lr_0.001-trainloss_0.6045-evalloss_0.1127-evalpsnr_30.2671.pkl', map_location='cuda:%d'%(gpu)))
    # SepConvNet.load_state_dict(torch.load('beta_LSTM_iter8-ltype_fSATD_fs-lr_0.001-trainloss_0.557-evalloss_0.1165-evalpsnr_29.8361.pkl'))
            
    # @@@ Test result: child from 0-27 is the raw model~
    grad_list = [18,19,20,21,  25,26,27]
    child_cnt = 0
    for child in SepConvNet.children():
        if child_cnt in grad_list:
            child_cnt += 1
            continue
        child_cnt += 1
        for param in child.parameters():
            param.requires_grad = False

    # cs = list(SepConvNet.children())
    # ps = list(cs[17].parameters())
    # IPython.embed()
    # exit()
    # MSE_cost = nn.MSELoss().cuda()
    # SepConvNet_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    # SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_optimizer = optim.Adamax(filter(lambda p: p.requires_grad, SepConvNet.parameters()),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True)

    # ----------------  Time part -------------------
    start_time = time.time()
    global_step = 0
    # ----------------  Time part -------------------


    # ---------------- Opt part -----------------------
    # opter = Opter(gpu)
    # -------------------------------------------------
    # print('[!] Ready to train!')
    # IPython.embed()

    for epoch in range(0,EPOCH):
        SepConvNet.train().cuda()
        cnt = 0
        sumloss = 0.0 # The sumloss is for the whole training_set
        tsumloss = 0.0 # The tsumloss is for the printinterval

        sumloss_b = 0.0 # The sumloss is for the whole training_set
        tsumloss_b = 0.0 # The tsumloss is for the printinterval

        printinterval = 500
        print("---------------[Epoch%3d]---------------"%(epoch + 1))
        for label_list in trainLoader:
            bad_list = label_list[7:]
            label_list = label_list[:7]
            # IPython.embed()
            # exit()
            global_step = global_step + 1
            cnt = cnt + 1

            for i in range(5):
                imgL = var(bad_list[i]).cuda()
                imgR = var(bad_list[i+1]).cuda()
                poor_label = var(bad_list[i+2]).cuda()

                label = var(label_list[i+2]).cuda()
                label_L = var(label_list[i]).cuda()

                # ----------- Forward prediction -----------
                SepConvNet_optimizer.zero_grad()
                if i == 0:
                    output_f, stat = SepConvNet(imgL, imgR, 0)
                else:
                    output_f, stat = SepConvNet(imgL, imgR, 0, res_c, stat)

                loss = SepConvNet_cost(output_f, label)

                loss.backward(retain_graph=True)
               

                sumloss = sumloss + loss.data.item()
                tsumloss = tsumloss + loss.data.item()


                # ----------- Backward prediction -----------
                SepConvNet_optimizer.zero_grad()
                
                output_b, stat = SepConvNet(output_f, imgR, 1, tensorHidden=stat)

                loss = SepConvNet_cost(output_b, label_L)
                if i < 4:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=False)

                sumloss_b = sumloss_b + loss.data.item()
                tsumloss_b = tsumloss_b + loss.data.item()

                res_f = poor_label - output_f
                res_b = imgL - output_b
                res_c = torch.cat([res_f, res_b], 1)
            
            
            if cnt % printinterval == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Time [%4.4f], Back loss[%.6f], Interval loss [%.6f]' %
                    (epoch + 1, EPOCH, cnt, len(trainset) // BATCH_SIZE, time.time() - start_time, tsumloss_b / printinterval / 5, tsumloss / printinterval / 5))
                tsumloss = 0.0
                tsumloss_b = 0.0
        print('Epoch [%d/%d], iter: %d, Time [%4.4f], Avg Loss [%.6f]' %
            (epoch + 1, EPOCH, cnt, time.time() - start_time, sumloss / cnt / 5))


        # ---------------- Part for validation ----------------
        trainloss = sumloss / cnt
        SepConvNet.eval().cuda()
        evalcnt = 0
        pos = 0.0
        sumloss = 0.0
        sumloss_b = 0.0
        psnr = 0.0
        psnr_b = 0.0
        for label_list in valLoader:
            
            bad_list = label_list[7:]
            label_list = label_list[:7]
            loss_s = []
            with torch.no_grad():
                for i in range(5):
                    imgL = var(bad_list[i]).cuda()
                    imgR = var(bad_list[i+1]).cuda()
                    poor_label = var(bad_list[i+2]).cuda()

                    label = var(label_list[i+2]).cuda()
                    label_L = var(label_list[i]).cuda()

                    # ----------- Forward prediction -----------
                    if i == 0:
                        output_f, stat = SepConvNet(imgL, imgR, 0)
                    else:
                        output_f, stat = SepConvNet(imgL, imgR, 0, res_c, stat)
                    loss = SepConvNet_cost(output_f, label)

                    psnr = psnr + calcPSNR.calcPSNR(output_f.cpu().data.numpy(), label.cpu().data.numpy())
                    sumloss = sumloss + loss.data.item()
                    # sumloss = sumloss + loss.data.item()
                    # tsumloss = tsumloss + loss.data.item()


                    # ----------- Backward prediction -----------
                    output_b, stat = SepConvNet(output_f, imgR, 1, tensorHidden=stat)

                    loss_b = SepConvNet_cost(output_b, label_L)
                    psnr_b = psnr + calcPSNR.calcPSNR(output_b.cpu().data.numpy(), label_L.cpu().data.numpy())
                    sumloss_b = sumloss_b + loss_b.data.item()
                    # sumloss_b = sumloss_b + loss.data.item()
                    # tsumloss_b = tsumloss_b + loss.data.item()

                    res_f = poor_label - output_f
                    res_b = imgL - output_b
                    res_c = torch.cat([res_f, res_b], 1)


                evalcnt = evalcnt + 5

        # ------------- Tensorboard part -------------
        # writer.add_scalar("Valid SATD loss", sumloss / evalcnt, epoch)
        # writer.add_scalar("Valid PSNR", psnr / valset.__len__(), epoch)
        # ------------- Tensorboard part -------------
        print('Validation loss [%.6f],  Average PSNR [%.4f], [!] Backward loss [%.6f] PSNR[%.4f]' % (
        sumloss / evalcnt, psnr / evalcnt, sumloss_b / evalcnt, psnr_b / evalcnt))
        SepConvNet_schedule.step(psnr / evalcnt)
        torch.save(SepConvNet.state_dict(),
                os.path.join('.', 'test_share_dual_LSTM_iter' + str(epoch + 1)
                                + '-ltype_fSATD_fs'
                                + '-lr_' + str(LEARNING_RATE)
                                + '-trainloss_' + str(round(trainloss, 4))
                                + '-evalloss_' + str(round(sumloss / evalcnt, 4))
                                + '-evalpsnr_' + str(round(psnr / evalcnt, 4)) + '.pkl'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
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
