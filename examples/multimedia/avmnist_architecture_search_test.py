from unimodals.common_models import LeNet, MLP, Constant
from training_structures.architecture_search import train, test
import utils.surrogate as surr
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '/mnt/e/Laboratory/datasets/AV_MNIST', batch_size=32)
model = torch.load('temp/best.pt').cuda()
test(model, testdata, no_robust=True)
