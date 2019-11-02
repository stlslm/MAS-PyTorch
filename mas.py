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

from optimizer_lib import *
from model_train import *


def mas_train(model, task_no, no_of_layers, no_of_classes, dataloader, dset_size, use_gpu = False):
	
	"""
	Inputs:
	1) model: A reference to the model that is being exposed to the data for the task
	2) task_no: The task that is being exposed to the model identified by it's number
	3) no_of_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
	4) no_of_classes: The number of classes in the task  
	5) dataloader: Dataloader that feeds data to the model
	6) dset_size: The size of the task (size of the dataset belonging to the task)
	7) use_gpu: Set the flag to `True` if you want to train the model on GPU

	Outputs:
	1) model: Returns a trained model

	Function: Trains the model on a particular task and deals with different tasks in the sequence
	"""

	#this is the task to which the model is exposed
	if (t == 1):
		#initialize the reg_params for this task
		model, freeze_layers = create_freeze_layers(model, no_of_layers)
		model.reg_params = init_reg_params(model, use_gpu, freeze_layers)

	else:
		#inititialize the reg_params for this task
		model.reg_params = init_reg_params_across_tasks(model, use_gpu)

	#get the optimizer
	optimizer_sp = local_sgd(model.tmodel.parameters(), lr = 0.001)

	model = train_model(model, path, optimizer_sp, model_criterion, dataloader, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003)

	if (t > 1):
		model = consolidate_reg_params(model, use_gpu)

	return model






