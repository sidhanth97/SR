from imports import *
import model
from perception_layer import VGGFeat


batch_size = 25
img_size = 64
img_channels = 3
n_examples = 2000
epochs = 10
lr = 0.0002
beta = 0.5
scale = 2

train_dataset = '/home/administrator/Sidhanth/Datasets/sr_dataset/Flickr/flickr30k_images/flickr30k_images/*.jpg'

x_size = (img_size,img_size)
y_size = (img_size*scale,img_size*scale)

def load(dataset,n_examples=n_examples):
	count = 0
	
	x_image = []
	y_image = []
	for instance in tqdm(sorted(glob.glob(dataset))):
		image = Image.open(instance)
		image_lr = transform_fn(img_size)(image)
		image_hr = transform_fn(img_size,scale=scale)(image)	
		#print(image_lr.shape,image_hr.shape)
		x_image.append(image_lr)
		y_image.append(image_hr)

		count+=1
		if count==n_examples:
			break

	x_image = torch.stack(x_image).float()
	y_image = torch.stack(y_image).float()
	print('Images loaded with shapes:',x_image.shape,y_image.shape)
	return x_image,y_image


def transform_fn(img_size,scale=1):
	transform = transforms.Compose([transforms.Resize(img_size*scale),
									transforms.CenterCrop(img_size*scale),
									transforms.ToTensor()])
	return transform



ngpu = 1
device = torch.device('cuda:0' if(torch.cuda.is_available() and ngpu>0) else 'cpu')
x,y = load(train_dataset)

writer = SummaryWriter()
netG = model.Generator(img_size,writer=writer).cuda()
netD = model.Discriminator(img_size*scale,writer=writer).cuda()
vgg_feat = VGGFeat().cuda()

criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()
optimizerG = optim.AdamW(netG.parameters(),lr=lr,betas=(beta,0.999))
optimizerD = optim.AdamW(netD.parameters(),lr=lr,betas=(beta,0.999))

val = int(n_examples/batch_size)
count = 0
real = 1
fake = 0

for epoch in tqdm(range(epochs), desc='Epoch',total=epochs):
	avg_loss = 0.0	
	gen_cost = 0
	dis_cost = 0
	start_index = 0
	end_index = 0
	dis_count = 0
	gen_count = 0
	with tqdm(total=val) as pbar:
		try:
			while end_index<n_examples:
				end_index = start_index + batch_size

				image_lr = x[start_index:end_index].to(device)
				image_hr = y[start_index:end_index].to(device)
				gen_image = netG(image_lr)

				real_label = torch.full((batch_size,),real,device=device)
				fake_label = torch.full((batch_size,),fake,device=device)

				# Discriminator Training
				netD.zero_grad()
				real_output = netD(image_hr).view(-1)
				errD_real = criterion_bce(real_output,real_label)
				D_x = real_output.mean().item()
				fake_output = netD(gen_image.detach()).view(-1)
				errD_fake = criterion_bce(fake_output,fake_label)
				D_G_x = fake_output.mean().item()
				errD = errD_fake + errD_real
				dis_cost += errD.item()/val
				errD.backward()
				optimizerD.step()

				netG.zero_grad()
				real_feat = vgg_feat(image_hr)
				fake_feat = vgg_feat(gen_image)
				errG_reconstruction = criterion_mse(fake_feat,real_feat) 
				gen_output = netD(gen_image).view(-1)
				errG_adversarial = criterion_bce(gen_output,real_label)
				errG = errG_reconstruction + (1e-3)*errG_adversarial
				gen_cost += errG.item()/val
				errG.backward()
				optimizerG.step()

				if D_x < 0.1 or D_G_x > 0.9:
					optimizerD.step()
					dis_cost += 1

				if D_G_x < 1e-2 or D_x > 0.99:
					optimizerG.step()
					gen_count += 1
								
				# Summaries
				writer.add_images('Input',image_lr)
				writer.add_images('Generator',gen_image)
				writer.add_images('Label',image_hr)
				writer.add_scalar('Discriminator Training Loss',dis_cost,count)
				writer.add_scalar('Generator Training Loss',gen_cost,count)
				writer.add_scalar('D(x)',D_x,count)
				writer.add_scalar('D(G(x))',D_G_x,count)
				count += 1
				dis_cost += 1
				gen_count += 1
				start_index = end_index
				pbar.set_postfix_str(s='D_loss:'+str(errD.item())+' G_loss:'+str(errG.item())+' D(x):'+str(D_x)+' D(G(x)):'+str(D_G_x))
				pbar.update()
		except KeyboardInterrupt:
			print('Done')

	print('(D_count,G_count):',dis_count,gen_count)


"""
real_batch = next(iter(img_dataloader))
label_batch = next(iter(label_dataloader))


plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Label Images')
plt.imshow(np.transpose(vutils.make_grid(label_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

plt.show()

"""

"""

workers = 2
ngpu = 1




class TrainDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir,data_dir,img_size):
		self.root_dir = root_dir
		self.examples = glob.glob(data_dir)
		self.img_size = img_size

	def __len__(self):
		return len(self.examples)

	def __getitem__(self,index):
		img_name = self.examples[index]
		image = Image.open(img_name)
		image_lr = transform_fn(self.img_size)(image)
		image_hr = transform_fn(self.img_size,scale=4)(image)
		sample = {'image_lr':image_lr,'image_hr':image_hr}
		return sample


dataset = TrainDataset(dataroot,dataroot+'/flickr30k_images/flickr30k_images/*.jpg',img_size)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=workers)
"""



"""
img_dataset = dset.ImageFolder(root=dataroot,
						   transform=transforms.Compose([
						   transforms.Resize(img_size),
						   transforms.CenterCrop(img_size),
						   transforms.ToTensor()
						  ]))

label_dataset = dset.ImageFolder(root=dataroot,
						   transform=transforms.Compose([
						   transforms.Resize(img_size*scale),
						   transforms.CenterCrop(img_size*scale),
						   transforms.ToTensor()
						  ]))

img_dataloader = torch.utils.data.DataLoader(img_dataset,batch_size=batch_size,num_workers=workers)
label_dataloader = torch.utils.data.DataLoader(label_dataset,batch_size=batch_size,num_workers=workers)
"""
