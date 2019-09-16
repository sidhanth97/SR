from imports import *


def act(act_type,**kwargs):
    if act_type == 'relu':
        layer = nn.ReLU(True,**kwargs)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,True,**kwargs)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=1,init=0.25,**kwargs)
    elif act_type == 'tanh':
        layer = nn.Tanh()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()

    return layer


def norm(norm_type,nc):
	if norm_type == 'batch':
		layer = nn.BatchNorm2d(nc,affine=True)
	elif norm_type == 'instance':
		layer = nn.InstanceNorm2d(nc,affine=False)
	return layer


def pad(pad_type, kernel_size=None, exact_pad_size=None):
    if kernel_size:
        pad_size = (kernel_size - 1) // 2
    if exact_pad_size:
        pad_size = exact_pad_size
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(pad_size)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(pad_size)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(pad_size)
    return layer


def dropout(p=0.2):
    return nn.Dropout(p)


def identity(input):
    return input


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.submodule = submodule

    def forward(self, input):
        output = input + self.submodule(input)
        return output

    def __repr__(self):
        tmpstr =  'Identity + \n|'
        modstr = self.submodule.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class Conv2dBlock(nn.Module):
    def __init__(self,name,input_nc,output_nc,kernel_size,stride=1,dilation=1,groups=1,bias=True,pad_type='zero',norm_type=None,act_type='prelu',use_dropout=False,writer=None):
        super(Conv2dBlock, self).__init__()
        self.P = pad(pad_type, kernel_size) if pad_type else identity
        self.C = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.N = norm(norm_type, output_nc) if norm_type else identity
        self.A = act(act_type) if act_type else identity
        self.D = dropout() if use_dropout else identity
        self.weight_init()
        self.writer = writer
        self.name = name

    def forward(self, input):
        output = self.P(input)
        output = self.C(output)
        output = self.N(output)
        output = self.A(output)
        output = self.D(output)
        if self.writer is not None:
            self.writer.add_histogram(self.name+'_weight',self.C.weight)
            self.writer.add_histogram(self.name+'_bias',self.C.bias)
        return output

    def weight_init(self):
        nn.init.kaiming_normal_(self.C.weight,a=1.0,mode='fan_in')
        if self.C.bias is not None:
            self.C.bias.data.zero_()
        if isinstance(self.N,nn.BatchNorm2d):
            self.N.weight.data.fill_(1)
            self.N.bias.data.zero_()


class DenseBlock(nn.Module):
    def __init__(self,name,input_dim,output_dim,drop_out=True,act_type='lrelu',writer=None):
        super(DenseBlock,self).__init__()
        self.weight_tensor = torch.empty((input_dim,output_dim),requires_grad=True)
        self.bias_tensor = torch.zeros((output_dim),requires_grad=True)
        nn.init.kaiming_normal_(self.weight_tensor,a=1.0,mode='fan_in')
        self.weight_tensor = self.weight_tensor.cuda()
        self.bias_tensor = self.bias_tensor.cuda()
        self.A = act(act_type) if act_type else identity
        self.D = dropout() if drop_out else identity
        self.writer = writer
        self.name = name
    
    def forward(self,input):
        output = torch.matmul(input,self.weight_tensor)
        output = self.A(output+self.bias_tensor)
        if self.writer is not None:
            self.writer.add_histogram(self.name+'_weight',self.weight_tensor)
            self.writer.add_histogram(self.name+'_bias',self.bias_tensor)
        return output         


def ResNetBlock(name,input_nc,mid_nc,output_nc,kernel_size=3,stride=1,bias=True,pad_type='zero',norm_type='batch',act_type='prelu',use_dropout=False,writer=None):
    conv1 = Conv2dBlock(name+'_res_block_1',input_nc,mid_nc,kernel_size,stride,bias=bias,pad_type=pad_type,norm_type=norm_type,act_type=act_type,use_dropout=use_dropout,writer=writer)
    conv2 = Conv2dBlock(name+'_res_block_2',mid_nc,output_nc,kernel_size,stride,bias=bias,pad_type=pad_type,norm_type=norm_type,act_type=None,use_dropout=False,writer=writer)
    residual_features = nn.Sequential(conv1, conv2)
    return ShortcutBlock(residual_features)


class SubPixelConvBlock(nn.Module):
    def __init__(self,name,input_nc,output_nc,upscale_factor=2,kernel_size=3,stride=1,bias=True,pad_type='zero',norm_type=None,act_type='prelu',use_dropout=False,writer=None):
        super(SubPixelConvBlock, self).__init__()
        self.conv_block = Conv2dBlock(name+'_sub_pixel_block',input_nc,output_nc * (upscale_factor ** 2),kernel_size,stride,bias=bias,pad_type=pad_type,norm_type=norm_type,act_type=act_type,use_dropout=use_dropout,writer=writer)
        self.PS = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        output = self.conv_block(input)
        output = self.PS(output)
        return output

