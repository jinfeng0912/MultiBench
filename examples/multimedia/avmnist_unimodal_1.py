import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.unimodal import train, test


modalnum = 1
traindata, validdata, testdata = get_dataloader(
    '/mnt/e/Laboratory/datasets/AV_MNIST',
    batch_size=16,
    num_workers=0,
    max_train=12000,
    max_test=3000)
channels = 6
# encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
encoder = LeNet(1, channels, 5).cuda()
head = MLP(channels*32, 100, 10).cuda()


train(encoder, head, traindata, validdata, 10, optimtype=torch.optim.SGD,
      lr=0.1, weight_decay=0.0001, modalnum=modalnum,
      save_encoder='avmnist_unimodal_1_encoder.pt', save_head='avmnist_unimodal_1_head.pt')

print("Testing:")
encoder = torch.load('avmnist_unimodal_1_encoder.pt').cuda()
head = torch.load('avmnist_unimodal_1_head.pt')
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
