from imports import *

class VGGFeat(nn.Module):
	def __init__(self, n_layers=16, use_bn=False, i_max_pool=0, include_max_pool=False):
		super(VGGFeat, self).__init__()
		if n_layers == 16:
			if use_bn:
				model = torchvision.models.vgg16_bn(pretrained=True)
			else:
				model = torchvision.models.vgg16(pretrained=True)
		elif n_layers == 19:
			if use_bn:
				model = torchvision.models.vgg19_bn(pretrained=True)
			else:
				model = torchvision.models.vgg19(pretrained=True)

		self.features = self.__break_layers(model.features, i_max_pool, include_max_pool)
		self.mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda() 
		self.std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()

		for param in model.parameters():
			param.requires_grad = False


	def __break_layers(self, features, i, include_max_pool=False):
		children = list(features.children())
		max_pool_indices = [index for index, m in enumerate(children) if isinstance(m, nn.MaxPool2d)]
		target_features = children[:max_pool_indices[i]] if include_max_pool else children[:max_pool_indices[i]-1]
		return nn.Sequential(*target_features)

	def forward(self, input):
		input = (input - self.mean) / self.std
		output = self.features(input)
		return output

