
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_class import *
from optimizer_lib import *
from utils.mas_utils import init_reg_params, compute_omega_grads_norm_no_optim, init_reg_params_across_tasks_no_tensorkey, consolidate_reg_params_no_tensorkey

##############
batch_size =32
use_gpu = True

dloaders_train = []
data_path = os.path.join(os.getcwd(), "Data")

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),

	'test': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}
data_dir = os.path.join(os.getcwd(), "Data")
num_classes = []
for tdir in sorted(os.listdir(data_dir)):

	#create the image folders objects
	tr_image_folder = datasets.ImageFolder(os.path.join(data_dir, tdir, "train"), transform = data_transforms['train'])
	te_image_folder = datasets.ImageFolder(os.path.join(data_dir, tdir, "test"), transform = data_transforms['test'])

	#get the dataloaders
	tr_dset_loaders = torch.utils.data.DataLoader(tr_image_folder, batch_size=batch_size, shuffle=True, num_workers=4)
	te_dset_loaders = torch.utils.data.DataLoader(te_image_folder, batch_size=batch_size, shuffle=True, num_workers=4)

	#get the sizes
	temp1 = len(tr_image_folder) 
	temp2 = len(te_image_folder)


	#append the dataloaders of these tasks
	dloaders_train.append(tr_dset_loaders)

	#get the classes (THIS MIGHT NEED TO BE CORRECTED)
	num_classes.append(len(tr_image_folder.classes))


	#get the sizes array


def model_criterion(preds, labels):
	"""
	Function: Model criterion to train the model
	
	"""
	loss =  nn.CrossEntropyLoss()
	return loss(preds, labels)


task = 1
dataloader_train = dloaders_train[task-1]
device = torch.device("cuda:0")
reg_lambda = 0.01

pre_model = models.alexnet(pretrained = True)
model = shared_model(pre_model)

task_no = 0
for task_no, dataloader_train in enumerate(dloaders_train):
	task_no += 1

	# init per task
	no_of_classes = num_classes[task_no-1]

	# add a new head
	in_features = model.tmodel.classifier[-1].in_features
	del model.tmodel.classifier[-1]
	model.tmodel.classifier.add_module('6', nn.Linear(in_features, no_of_classes))

	if task_no == 1:
		model = init_reg_params(model, True)
	else:
		model = init_reg_params_across_tasks_no_tensorkey(model, True)

	# optimizer = local_sgd(model.reg_params, 0.001)
	optimizer = torch.optim.SGD(model.tmodel.parameters(), lr=0.001, momentum=0.9)

	for ep in range(1):

		# train for 1 epoch
		for data in dataloader_train:
			input_data, labels = data

			input_data = input_data.to(device)
			labels = labels.to(device) 

			model.tmodel.to(device)
			optimizer.zero_grad()

			output = model.tmodel(input_data)

			_, preds = torch.max(output, 1)
			loss = model_criterion(output, labels)

			loss.backward()

			# print('')
			i=0
			# for p in model.parameters():
			#	 i+=1
			#	 print(p.sum())
			#	 if i==10: break

			model = MAS_step(model, model.reg_params, reg_lambda)
			# print("")
			i=0
			# for p in model.parameters():
			#	 i+=1
			#	 print(p.sum())
			#	 if i==10: break
			# assert check_MAS_step(model_bef, model_aft)

			optimizer.step()

		# update omega
		# optimizer_ft = torch.optim.SGD(model.tmodel.parameters(), lr=0.001, momentum=0.9)   
		print ("Updating the omega values for this task")
		model = compute_omega_grads_norm_no_optim(model, dataloader_train, use_gpu)

	if (task_no > 1):
		model = consolidate_reg_params_no_tensorkey(model, use_gpu)


for group in optimizer.param_groups:
	for p in group['params']:
		if p in model.reg_params:
			print('gotcha')		

# for name, param in model.tmodel.named_parameters():
	# print(name)


d = {
		'a':3, 
		'b':4,
		'son': '77'
	}

print('son' in d)
print('ha' in d)	