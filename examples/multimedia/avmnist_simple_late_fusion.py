import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

traindata, validdata, testdata = get_dataloader(
    '/mnt/e/Laboratory/datasets/AV_MNIST',
    batch_size=32,
    num_workers=0,
    max_train=12000,
    max_test=3000)
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 10,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001,
      save='avmnist_simple_late_fusion_best.pt')
model = torch.load('avmnist_simple_late_fusion_best.pt').cuda()
test(model, testdata, no_robust=True)
