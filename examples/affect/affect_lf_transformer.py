import torch
import sys
import os
import torch.nn as nn
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from unimodals.common_models import Transformer, MLP  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import Concat  # noqa
from multiprocessing import freeze_support  # Windows 兼容

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules')

# Load data (GloVe sarcasm, classification for [B,1] collate)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm.pkl',
    robust_test=False, max_pad=True, task='classification')  # Binary sarcasm

# Simple CE: pool 3D Transformer seq + clamp/stabilize + long device
def simple_ce(pred, target, *args):
    if pred.dim() > 2: pred = pred.mean(1)  # [B,50,2] -> [B,2]
    pred = torch.clamp(pred, -10, 10)  # Avoid inf/NaN
    target = target.squeeze(-1).long().to(pred.device)  # [B,1] -> [B] 0/1
    return nn.CrossEntropyLoss()(pred, target)  # Binary [B,2] + [B]

# Sarcasm Transformer dims (GloVe 300 text -> 600 hidden, concat 1100)
encoders = [
    Transformer(371, 400).cuda(),  # Visual
    Transformer(81, 100).cuda(),   # Audio
    Transformer(300, 600).cuda()   # Text (GloVe 300, not BERT)
]
head = MLP(1100, 256, 2).cuda()  # 400+100+600=1100 -> binary 2

fusion = Concat().cuda()  # Late fusion [B,50,1100]

if __name__ == '__main__':
    freeze_support()

    train(encoders, fusion, head, traindata, validdata, 20, task="classification",  # Binary
          optimtype=torch.optim.AdamW, early_stop=False, is_packed=False,
          lr=1e-4, objective=simple_ce,  # CE pool fix
          save='sarcasm_lf_transformer_best.pt', weight_decay=0.01)

    print("Testing:")
    try:
        model = torch.load('sarcasm_lf_transformer_best.pt', weights_only=False).cuda()
    except FileNotFoundError:
        print("Warning: No best model (check loss for NaN) – test skipped")
        model = None
    if model:
        test(model=model, test_dataloaders_all=test_robust, dataset='sarcasm',  # Sarcasm binary
             is_packed=False, criterion=simple_ce,
             task='classification', no_robust=True, AUPRC=False)  # Acc/F1 binary
    else:
        print("Test skipped – rerun to save model")