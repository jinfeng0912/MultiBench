import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa


# MODIFIED: Changed data path to your mosei_senti_data.pkl
traindata, validdata, testdata = get_dataloader(
    '/mnt/e/Laboratory/datasets/CMU_MOSEI/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei
encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
head = Sequential(GRU(409, 512, dropout=True, has_padding=True,
                  batch_first=True, last_only=True), MLP(512, 512, 1)).cuda()

# humor/sarcasm
# encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
# head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True, last_only=True), MLP(1128, 512, 1)).cuda()

fusion = ConcatEarly().cuda()

# Use packed inputs because head GRU has has_padding=True; otherwise lists hit .float()
# train(encoders, fusion, head, traindata, validdata, 10, task="regression", optimtype=torch.optim.AdamW,
#       is_packed=True, lr=1e-3, save='mosei_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
# MODIFIED: Changed loaded model name
model = torch.load('mosei_ef_best.pt').cuda()
# Align with training: packed inputs expected by GRU with has_padding=True
test(model, testdata, 'affect', is_packed=True,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)