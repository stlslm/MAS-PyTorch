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


def mas_train(model, no_of_tasks, path_to_datasets):

	#Need to train over tasks 
	for t in range(1, no_of_tasks+1):

		print ("The model is being trained on task {}".format(t))

		#initialize reg_params for task 0
		if (t == 1):
			model = shared_model(models.alexnet(pretrained = True))
			model.reg_params = init_reg_params(model, use_gpu)

		
		#initialize reg_params for tasks > 0 
		else:
			model = shared_model(models.alexnet(pretrained = True))
			model.load_state_dict(torch.load(path))
			model.reg_params = init_reg_params_across_tasks(model, use_gpu)

		train_model()



def mas_test():
