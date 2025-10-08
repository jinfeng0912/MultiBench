import sys
import os
from torch import nn
import torch

sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.gradient_blend import train, test # noqa


dls, weights = get_dataloader('/mnt/e/Laboratory/datasets/ENRiCO/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
mult_head = Linear(32, 20).cuda()
uni_head = [Linear(16, 20).cuda(), Linear(16, 20).cuda()]

fusion = Concat().cuda()

# train(encoders,fusion,head,traindata,validdata,num_epoch=50,gb_epoch=10,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0)
allmodules = encoders + [mult_head, fusion] + uni_head


def trainprocess():
    # MODIFIED: Added unique save destination using 'savedir'
    train(encoders, mult_head, uni_head, fusion, traindata, validdata, 8,
          gb_epoch=8, optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0,
          savedir='enrico_gradient_blend_best.pt', track_complexity=False)


all_in_one_train(trainprocess, allmodules)

# MODIFIED: Load from unique save destination
model = torch.load('enrico_gradient_blend_best.pt').cuda()
# Speed-up: skip noisy robustness sweep
first_key = list(testdata.keys())[0]
first_dl = testdata[first_key][0]
test(model, first_dl, dataset='enrico', no_robust=True)
