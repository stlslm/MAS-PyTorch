from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil

import sys
sys.path.append('utils')

from model_utils import *
from mas_utils import *


def train_model(model, path, optimizer, model_criterion, dataloader, dset_size, num_epochs, use_gpu = False, lr = 0.001):
	"""
	Inputs:
	1) model: A reference to the model that is being exposed to the data for the task
	2) optimizer: A local_sgd optimizer object that implements the idea of MaS
	3) model_criterion: The loss function used to train the model
	4) dataloader: A dataloader to feed the data to the model
	5) dset_size: Size of the dataset that belongs to a specific task
	6) num_epochs: Number of epochs that you wish to train the model for
	7) use_gpu: Set the flag to `True` if you wish to train on a GPU. Default value: False
	8) lr: The initial learning rate set for training the model

	Outputs:
	1) model: Return a trained model

	Function: Trains the model on a specific task identified by a task number
	
	"""
	omega_epochs = num_epochs + 1

	store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
	model_path = os.path.join(os.getcwd(), "models")

	#create a models directory if the directory does not exist
	if (task_no == 1 and not os.path.isdir(model_path)):
		os.mkdir(path_to_model)

	#the flag indicates that the the directory exists
	checkpoint_file, flag = check_checkpoint(store_path)

	if (flag == False):
		#create a task directory where the checkpoint files and the classification head will be stored
		create_task_dir(task_no, no_classes, store_path)
		start_epoch = 0

	else:
		
		####################### Get the checkpoint if it exists ###############################		
		
		#check for a checkpoint file	
		if (checkpoint_file == ""):
			start_epoch = 0

		else:
			print ("Loading checkpoint '{}' ".format(checkpoint_file))
			checkpoint = torch.load(checkpoint_file)
			start_epoch = checkpoint['epoch']
			
			print ("Loading the model")
			model = shared_model(models.alexnet(pretrained = True))
			model = model.load_state_dict(checkpoint['state_dict'])
			
			print ("Loading the optimizer")
			optimizer = local_sgd(model.reg_params)
			optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
			
			print ("Done")

		######################################################################################

	
	#commencing the training loop
	for epoch in range(start_epoch, omega_epochs):			
		
		#run the omega accumulation at convergence of the loss function
		if (epoch == omega_epochs -1):
			#no training of the model takes place in this epoch
			optimizer_ft = omega_update(model.reg_params)
			print ("Updating the omega values for this task")
			model = compute_omega_grads_norm(model, dataloader, optimizer_ft)
		
		else:

			since = time.time()
			best_perform = 10e6

			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)
			print ("The {}ing phase is ongoing".format(phase))
			
			running_loss = 0
			
			#scales the optimizer every 20 epochs 
			optimizer = exp_lr_scheduler(optimizer, epoch, lr)

			model.train(True)


			for data in dset_loaders:
				input_data, labels = data

				del data

				if (use_gpu):
					input_data = input_data.to(device)
					labels = labels.to(device) 
				
				else:
					input_data  = Variable(input_data)
					labels = Variable(labels)
				
				model.to(device)
				optimizer.zero_grad()
				
				output = model.tmodel(input_data)
				del input_data

				_, preds = torch.max(outputs, 1)
				loss = model_criterion(output, labels)
				del output
		
				loss.backward()
				optimizer.step(model.reg_params)
		
				running_loss += loss.item()
				del loss

				running_corrects += torch.sum(preds == labels.data)
				del preds
				del labels

			epoch_loss = running_loss/dset_size
			epoch_accuracy = running_corrects.double()/dset_size


			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			
			#avoid saving a file twice
			if(epoch != 0 and epoch != num_epochs -1 and (epoch+1) % 10 == 0):
				epoch_file_name = os.path.join(store_path, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'epoch_accuracy': epoch_accuracy, 
				'model_state_dict': model_init.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


	#save the model and the performance 
	save_model(model, task_no, no_of_classes, epoch_accuracy)