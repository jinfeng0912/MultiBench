import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import Transformer, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa

# MODIFIED: Changed data path to mosei_senti_data.pkl
traindata, validdata, test_robust = \
    get_dataloader('/mnt/e/Laboratory/datasets/CMU_MOSEI/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei - Dimensions are assumed to be compatible
encoders = [Transformer(35, 40).cuda(),
            Transformer(74, 10).cuda(),
            Transformer(300, 600).cuda()]
head = MLP(650, 256, 1).cuda()

fusion = Concat().cuda()

# MODIFIED: Changed saved model name to mosei and made it more specific
train(encoders, fusion, head, traindata, validdata, 15, task="regression", optimtype=torch.optim.AdamW,
      early_stop=True, is_packed=True, lr=1e-4, save='mosei_lf_transformer_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


print("Testing:")
# MODIFIED: Changed loaded model name and test dataset name
model = torch.load('mosei_lf_transformer_best.pt').cuda()
test(model=model, test_dataloaders_all=test_robust, dataset='mosei', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)