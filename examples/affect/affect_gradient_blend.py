from torch import nn
import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))



from private_test_scripts.all_in_one import all_in_one_train # noqa
import training_structures # noqa
from training_structures.gradient_blend import train, test # noqa
from unimodals.common_models import GRU, MLP, Transformer # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl

# MODIFIED: Changed data path to your humor dataset
# MODIFIED: Added the data_type='humor' argument
traindata, validdata, test_robust = \
    get_dataloader('/mnt/e/Laboratory/datasets/UR_Funny_humor/humor.pkl',
                   task='classification', robust_test=False, max_pad=True, data_type='humor')

# mosi/mosei (DEACTIVATED)
# encoders = [Transformer(35, 70).cuda(),
#             Transformer(74, 150).cuda(),
#             Transformer(300, 600).cuda()]
# head = MLP(820, 512, 2).cuda()
#
# unimodal_heads = [MLP(70, 32, 2).cuda(), MLP(
#     150, 64, 2).cuda(), MLP(600, 256, 2).cuda()]

# humor/sarcasm (ACTIVATED)
encoders=[Transformer(371,700).cuda(), \
    Transformer(81,150).cuda(),\
    Transformer(300,600).cuda()]
head=MLP(1450,512,2).cuda()

unimodal_heads=[MLP(700,512,2).cuda(),MLP(150,64,2).cuda(),MLP(600,256,2).cuda()]

fusion = Concat().cuda()

# training_structures.gradient_blend.criterion = nn.L1Loss()

# MODIFIED: Changed saved model name to humor
# IMPORTANT: num_epoch must be >= gb_epoch, otherwise no training occurs and no model is saved.
# Changed gb_epoch from 20 -> 5 to ensure the training loop runs (15 // 5 = 3 cycles).
train(encoders, head, unimodal_heads, fusion, traindata, validdata, 10, gb_epoch=5, lr=1e-3, AUPRC=False,
      classification=True, optimtype=torch.optim.AdamW, savedir='humor_gb_best.pt', weight_decay=0.1)

print("Testing:")
# MODIFIED: Changed loaded model name to humor and fixed typo (besf -> best)
model = torch.load('humor_gb_best.pt').cuda()

# MODIFIED: Changed dataset name to humor
test(model, test_robust, dataset='humor', auprc=False, no_robust=True)

# test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')