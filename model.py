from imports import *
import utils as U

class Generator(nn.Module):
	def __init__(self,img_size,input_ngc=3,output_ngc=3,ngf=128,ngb=2,n_upscale=1,norm_type='batch',act_type='prelu',writer=None):
		super(Generator,self).__init__()

		first_conv = U.Conv2dBlock('gen_conv_1',input_ngc, ngf, kernel_size=3, norm_type=None, act_type=act_type,writer=writer)
		resnet_blocks = [U.ResNetBlock('gen_conv_'+str(i + 1),ngf, ngf, ngf, norm_type=norm_type, act_type=act_type,writer=writer) for i in range(ngb)]
		before_up_conv = U.Conv2dBlock('gen_conv_penultima',ngf, ngf, kernel_size=3, norm_type=norm_type, act_type=act_type,writer=writer)

		self.features_LR = nn.Sequential(first_conv,U.ShortcutBlock(nn.Sequential(*resnet_blocks, before_up_conv)))
		self.features_HR = nn.Sequential(*[U.SubPixelConvBlock('gen_Upscaling',ngf, ngf, upscale_factor=2, kernel_size=3, norm_type=None, act_type=act_type,writer=writer) for _ in range(n_upscale)])
		self.reducer = U.Conv2dBlock('gen_final_conv',ngf, output_ngc, kernel_size=3, norm_type=None, act_type='tanh',writer=writer)

		self.mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
		self.std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()

	def forward(self, input):
		input = (input-self.mean)/self.std
		features_LR = self.features_LR(input)
		features_HR = self.features_HR(features_LR)
		output = self.reducer(features_HR)
		return output


class Discriminator(nn.Module):
	def __init__(self,img_size,input_ndc=3,output_ndc=1,filter_num=32,dense_num=1024,int_blocks=4,norm_type='batch',act_type='lrelu',writer=None):
		super(Discriminator,self).__init__()

		first_conv = [U.Conv2dBlock('dis_conv_1',input_ndc,filter_num,kernel_size=3,norm_type=None,act_type=act_type,writer=writer)]
		first_conv += [U.Conv2dBlock('dis_conv_2',filter_num,filter_num,kernel_size=3,norm_type=norm_type,act_type=act_type,writer=writer)]

		internal_blocks = []
		for i in range(int_blocks):
			internal_blocks += [U.Conv2dBlock('dis_int_conv_'+str(i+1),filter_num*(2**i),filter_num*(2**(i+1)),kernel_size=3,norm_type=norm_type,act_type=act_type,writer=writer)]
			internal_blocks += [U.Conv2dBlock('dis_int_conv_stride_'+str(i+1),filter_num*(2**(i+1)),filter_num*(2**(i+1)),kernel_size=3,stride=2,norm_type=None,act_type=None,writer=writer)]
		self.features = nn.Sequential(*first_conv,*internal_blocks)

		dense_block = [nn.Flatten()]
		vector_dimension = int((img_size*img_size*filter_num)/(2**int_blocks))
		dense_block += [U.DenseBlock('dis_dense_1',vector_dimension,dense_num,writer=writer)]
		dense_block += [U.DenseBlock('dis_dense_2',dense_num,output_ndc,drop_out=False,act_type='sigmoid',writer=writer)]
		self.reducer = nn.Sequential(*dense_block)

		self.mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
		self.std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()

	def forward(self,input):
		input = (input-self.mean)/self.std
		output = self.features(input)
		output = self.reducer(output)
		return output
