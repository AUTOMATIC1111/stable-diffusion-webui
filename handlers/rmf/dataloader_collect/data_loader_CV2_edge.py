# data loader
from __future__ import print_function, division
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

#==========================dataset load==========================

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = cv2.resize(image,(self.output_size,self.output_size))
		lbl = cv2.resize(label,(self.output_size,self.output_size))
		edg = cv2.resize(edge,(self.output_size,self.output_size))

		return {'image':img,'label':lbl,'edge':edg}





class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		tocorp = np.random.randint(0,100)
		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		if tocorp < 50:
			top = np.random.randint(0, h - new_h)
			left = np.random.randint(0, w - new_w)

			image = image[top: top + new_h, left: left + new_w]
			label = label[top: top + new_h, left: left + new_w]
			edge = edge[top: top + new_h, left: left + new_w]

		else:
			image = cv2.resize(image,self.output_size)
			label = cv2.resize(label,self.output_size)
			edge = cv2.resize(edge,self.output_size)
		return {'image': image, 'label': label, 'edge':edge}


class RandomFlip(object):
	def __init__(self, p):
		self.prob = p

	def __call__(self, sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		toflip1 = np.random.randint(0,100)
		toflip2 = np.random.randint(0,100)
		# if toflip1 <=(100*self.prob):
		# 	image = image[::-1,:]
		# 	label = label[::-1,:] # up-down
		# 	edge = edge[::-1,:]
		if toflip2<=(100*self.prob):
			image = image[:,::-1]
			label = label[:,::-1] # left-right
			edge = edge[:,::-1]
		return {'image': image, 'label': label, 'edge':edge}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, sample):

		image, label, edge = sample['image'],sample['label'],sample['edge']
		tmpLbl = np.zeros(label.shape)
		tmpEdg = np.zeros(edge.shape)

		if(np.max(label)<1e-6):
			label = label
		elif(np.max(edge)<1e-6):
			edge = edge
		else:
			label = label/np.max(label)
			edge = edge/np.max(edge)


		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		image = image/np.max(image)
		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
		
		if len(tmpLbl.shape) == 2:
			tmpLbl[:,:] = label[:,:]
			tmpLbl = tmpLbl[:,:,np.newaxis]
		else:
			tmpLbl[:,:,0] = label[:,:,0]

		if len(tmpEdg.shape) == 2:
			tmpEdg[:,:] = edge[:,:]
			tmpEdg = tmpEdg[:,:,np.newaxis]
		else:
			tmpEdg[:,:,0] = edge[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = tmpLbl.transpose((2, 0, 1))
		tmpEdg = tmpEdg.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg.copy()),
			'label': torch.from_numpy(tmpLbl.copy()),
			'edge': torch.from_numpy(tmpEdg.copy())}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,edge_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.edge_name_list = edge_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = cv2.imread(self.image_name_list[idx])
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		original_shape = image.shape

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
			edge_3 = np.zeros(image.shape)
		else:
			label_3 = cv2.imread(self.label_name_list[idx])
			edge_3 = cv2.imread(self.edge_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		edge = np.zeros(edge_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(edge_3.shape)):
			edge = edge_3[:,:,0]
		elif(2==len(edge_3.shape)):
			edge = edge_3

		if 2==len(edge.shape):
			edge = edge[:,:,np.newaxis]

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis] #np.newaxis 增加一维

		sample = {'image':image, 'label':label, 'edge':edge,'original_shape':original_shape}

		if self.transform:
			sample = self.transform(sample)

		return {'image':sample['image'],'label':sample['label'],\
			'edge':sample['edge'],'original_shape':original_shape}
