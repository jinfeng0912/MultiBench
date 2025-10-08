import sys
import os
from torch import nn
import torch

sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from memory_profiler import memory_usage # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, LowRankTensorFusion # noqa
from training_structures.Supervised_Learning import train, test # noqa



dls, weights = get_dataloader('/mnt/e/Laboratory/datasets/ENRiCO/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
head = Linear(32, 20).cuda()

# fusion=Concat().cuda()
fusion = LowRankTensorFusion([16, 16], 32, 20).cuda()

allmodules = encoders + [head, fusion]


def trainprocess():
    # MODIFIED: Added unique save destination
    train(encoders, fusion, head, traindata, validdata, 8,
          optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0,
          save='enrico_low_rank_tensor_best.pt')


all_in_one_train(trainprocess, allmodules)

print("Testing:")
# MODIFIED: Load from unique save destination
model = torch.load('enrico_low_rank_tensor_best.pt').cuda()
# Speed-up: skip noisy robustness sweep
first_key = list(testdata.keys())[0]
first_dl = testdata[first_key][0]
test(model, first_dl, dataset='enrico', no_robust=True)
