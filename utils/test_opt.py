import torch
from Opt import Opter
import cv2
torch.cuda.set_device(0)
opt = Opter(0)


im1 = torch.from_numpy((cv2.imread('im1.png')[:,:,::-1].transpose(2, 0, 1) / 255.0).astype('float32')).unsqueeze(0)
im2 = torch.from_numpy((cv2.imread('im2.png')[:,:,::-1].transpose(2, 0, 1) / 255.0).astype('float32')).unsqueeze(0)

with torch.no_grad():
    flo = opt.estimate(im2, im1)
    warped = opt.warp(im1, flo).cpu().numpy()[0] * 255.0
    warped = warped.transpose(1,2,0).astype('uint8')[:,:,::-1]
    cv2.imwrite('im3.png', warped)