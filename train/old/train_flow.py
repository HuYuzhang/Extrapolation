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
import IPython

from network.FLOWNET import Network
from utils.RGB_Dataset import mydataset
from utils import correlation
from torch.nn.functional import grid_sample

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
Backward_tensorGrid = {}
Backward_tensorPartial = {}
def name2Tensor(f_name):
    tensorFirst = torch.FloatTensor(cv2.imread(f_name)[:128,:128,::-1].transpose(2, 0, 1).astype('float32') * (1.0 / 255.0)).view(1, 3, 128, 128).cuda()
    return tensorFirst
def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	if str(tensorFlow.size()) not in Backward_tensorPartial:
		Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
	tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

	tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

	tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

	return tensorOutput[:, :-1, :, :] * tensorMask


class PWCNet(torch.nn.Module):
	def __init__(self):
		super(PWCNet, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)), negative_slope=0.1, inplace=False)

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)
			# end
		# end

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()

		self.load_state_dict(torch.load('utils/network-default.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
		objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
		objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
		objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

		return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])


class Opter():
    def __init__(self, height, width, batch_size):
        self.pwc = PWCNet().cuda().eval()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        tensorHorizontal1 = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(batch_size, -1, height, -1)
        tensorVertical1 = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(batch_size, -1, -1, width)
        self.tensorAxisMap1 = torch.cat([ tensorHorizontal1, tensorVertical1 ], 1).cuda()

        tensorHorizontal2 = torch.linspace(-1.0, 1.0, width // 2).view(1, 1, 1, width // 2).expand(batch_size, -1, height // 2, -1)
        tensorVertical2 = torch.linspace(-1.0, 1.0, height // 2).view(1, 1, height // 2, 1).expand(batch_size, -1, -1, width // 2)
        self.tensorAxisMap2 = torch.cat([ tensorHorizontal2, tensorVertical2 ], 1).cuda()

        tensorHorizontal4 = torch.linspace(-1.0, 1.0, width // 4).view(1, 1, 1, width // 4).expand(batch_size, -1, height // 4, -1)
        tensorVertical4 = torch.linspace(-1.0, 1.0, height // 4).view(1, 1, height // 4, 1).expand(batch_size, -1, -1, width // 4)
        self.tensorAxisMap4 = torch.cat([ tensorHorizontal4, tensorVertical4 ], 1).cuda()

        # Here The tensorFirst/Second is of shape [bcz, C, H, W]
    def calcOpt(self, tensorFirst, tensorSecond):
        

        intHeight = tensorFirst.size(2)
        intWidth = tensorFirst.size(3)


        tensorPreprocessedFirst = tensorFirst
        tensorPreprocessedSecond = tensorSecond

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        

        tensorRawFlow = self.pwc(tensorPreprocessedFirst, tensorPreprocessedSecond)
        
        tensorFlow = 20.0 * torch.nn.functional.interpolate(input=tensorRawFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        tensorFlow[:, 0, :, :] *= 1 / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= 1 / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, flow, tensorFirst):
        '''
        Args:

        flow's shape:   [batch_size, 2, height, width]
        tensor's shape: [batch_size, 3, height, width]
        return shape:   [batch_size, 3, height, width]
        '''
        batch_size = tensorFirst.size(0)
        intHeight = tensorFirst.size(2)
        intWidth = tensorFirst.size(3)

        tensorHorizontal = torch.linspace(-1.0, 1.0, intWidth).view(1, 1, 1, intWidth).expand(batch_size, -1, intHeight, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, intHeight).view(1, 1, intHeight, 1).expand(batch_size, -1, -1, intWidth)
        tensorAxisMap = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        # if intWidth == 128:
        #     tensorAxisMap = self.tensorAxisMap1[:batch_size]
        # elif intWidth == 64:
        #     tensorAxisMap = self.tensorAxisMap2[:batch_size]
        # elif intWidth == 32:
        #     tensorAxisMap = self.tensorAxisMap4[:batch_size]
        # else:
        #     print("Error in warp size")
        #     exit()
        tensorFlowForWarp = (tensorAxisMap + flow).permute(0, 2, 3, 1)
        tensorWarped = grid_sample(tensorFirst, tensorFlowForWarp, mode='bilinear', padding_mode='zeros')
        
        return tensorWarped

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data,0.0)

def main(lr, batch_size, epoch, gpu, train_set, valid_set):
    


    # -------------- Some prepare ---------------------
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(gpu)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # -------------- Some prepare ---------------------

    BATCH_SIZE=batch_size
    EPOCH=epoch

    LEARNING_RATE = lr
    belta1 = 0.9
    belta2 = 0.999

    trainset = mydataset(train_set,transform_train)
    valset = mydataset(valid_set)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

    opter = Opter(128, 128, batch_size)

    SepConvNet = Network(opter).cuda()
    SepConvNet.apply(weights_init)
    dev_ = torch.device('cuda:'+str(gpu))
    SepConvNet.load_state_dict(torch.load('FLOW_iter7-ltype_fSATD_fs-lr_0.001-trainloss_0.1492-evalloss_0.1751-evalpsnr_25.0725.pkl', map_location=dev_))
    # IPython.embed()
    # exit()
    # SepConvNet_cost = nn.MSELoss().cuda()
    Opt_cost = nn.L1Loss().cuda()
    SepConvNet_cost = sepconv.SATDLoss().cuda()
    SepConvNet_optimizer = optim.Adamax(SepConvNet.parameters(),lr=LEARNING_RATE, betas=(belta1,belta2))
    SepConvNet_schedule = optim.lr_scheduler.ReduceLROnPlateau(SepConvNet_optimizer, factor=0.1, patience = 3, verbose=True, min_lr=1e-5)
    

    # ----------------  Time part -------------------
    start_time = time.time()
    global_step = 0
    # ----------------  Time part -------------------

	# ------------- Part for tensorboard --------------
    writer = SummaryWriter(comment="_FLOW")
	# ------------- Part for tensorboard --------------
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
            with torch.no_grad():
                # Remember here we need the back-forward flow
                diff = opter.calcOpt(imgR, imgL) 
                opt_label = opter.calcOpt(label, imgR)
            optPred, output = SepConvNet(diff, imgL, imgR)

            # loss = Opt_cost(optPred, opt_label)
            loss = SepConvNet_cost(output, label)
            loss.backward()
            
            SepConvNet_optimizer.step()

            # with torch.no_grad():
            #     loss = SepConvNet_cost(output, label)
            sumloss = sumloss + loss.data.item()
            tsumloss = tsumloss + loss.data.item()

            
                
            if cnt % printinterval == 0:
                writer.add_image("Target image", label[0], cnt)
                writer.add_image("Warped image", output[0], cnt)
                writer.add_image("Ref image", imgR[0], cnt)
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
                # Remember here we need the back-forward flow
                diff = opter.calcOpt(imgR, imgL)
                opt_label = opter.calcOpt(label, imgR)
            with torch.no_grad():
                optPred, output = SepConvNet(diff, imgL, imgR)
                loss = SepConvNet_cost(output, label)
                
                # loss = 0.5 * loss_out + 0.5 * loss_warp
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
                os.path.join('.', 'FLOW_iter' + str(epoch + 1)
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
