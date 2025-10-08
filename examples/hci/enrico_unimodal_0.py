import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned, VGG16Pruned # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa 
from memory_profiler import memory_usage # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.unimodal import train, test # noqa



dls, weights = get_dataloader('/mnt/e/Laboratory/datasets/ENRiCO/dataset')
traindata, validdata, testdata = dls
modalnum = 0
encoder = VGG11Slim(16, dropout=True, dropoutp=0.2,
                    freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]


def trainprocess():
    # MODIFIED: Added unique save destination
    train(encoder, head, traindata, validdata, 8, optimtype=torch.optim.Adam,
          lr=0.0001, weight_decay=0, modalnum=modalnum,
          save_encoder='enrico_unimodal_0_encoder.pt', save_head='enrico_unimodal_0_head.pt')


all_in_one_train(trainprocess, allmodules)

# MODIFIED: Load from unique save destination (encoder & head separately)
encoder = torch.load('enrico_unimodal_0_encoder.pt').cuda()
head = torch.load('enrico_unimodal_0_head.pt').cuda()
test(encoder, head, testdata, dataset='enrico', modalnum=modalnum)
