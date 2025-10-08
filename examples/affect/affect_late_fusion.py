import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch

# MODIFIED: Changed data path and type to mosei
traindata, validdata, test_robust = get_dataloader(
    '/mnt/e/Laboratory/datasets/CMU_MOSEI/mosei_raw.pkl', robust_test=False, data_type='mosei',batch_size=16)

# mosi/mosei - Dimensions are assumed to be compatible
encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(74, 200, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True, batch_first=True).cuda()]
head = MLP(870, 870, 1).cuda()

fusion = Concat().cuda()

# MODIFIED: Changed saved model name to mosei
train(encoders, fusion, head, traindata, validdata, 15, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save='mosei_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
# MODIFIED: Changed loaded model name and test dataset name
model = torch.load('mosei_lf_best.pt').cuda()
test(model=model, test_dataloaders_all=test_robust, dataset='mosei', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)