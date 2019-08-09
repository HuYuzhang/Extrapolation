'''
Note that this version derives from https://github.com/ndrplz/ConvLSTM_pytorch
Thanks to ndrplz for sharing his code.
Modified by Hu Yuzhang, to make it suitable for one-time-step forward in. 
Now it becomes more simple and sorry for that. 2019.8.9
'''
import torch.nn as nn
from torch.autograd import Variable
import torch


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

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


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
        if hidden_state is not None:
            # This means that what we use as the state is from the last state
        else:
            hidden_state = self.cell.init_hidden(batch_size=input_tensor.size(0))


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
        # ------------------- LSTM Part --------------------
            if tensorResidual is None:
                tensorResidual = Variable(torch.zeros(batch_size, tensorInput1.size(1), tensorInput1.size(2), tensorInput1.size(3))).cuda()
                tensorH_next, tensorC_next = self.moduleLSTM(tensorResidual) # Hence we also don't have the tensorHidden
            else:
                tensorH_next, tensorC_next = self.moduleLSTM(tensorResidual, tensorHidden)
            
        # ------------------- Encoder Part -----------------
            tensorJoin = torch.cat([ tensorInput1, tensorInput2, tensorh_next ], 1)
    
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