import torch
import sys
import os
import torch.nn as nn
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from unimodals.common_models import GRUWithLinear, MLP  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import TensorFusion  # noqa (no Concat for TF)

warnings.filterwarnings('ignore', category=UserWarning)

# Load data (sarcasm binary classification)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm.pkl',
    robust_test=False, max_pad=True, task='classification')

# Simple CE for binary (pool seq if 3D)
def simple_ce(pred, target, *args):
    if pred.dim() > 2: pred = pred.mean(1)
    pred = torch.clamp(pred, -10, 10)
    target = target.long().squeeze(-1).to(pred.device)
    return nn.CrossEntropyLoss()(pred, target)

# Encoders (sarcasm dims, no padding uniform len=50)
encoders = [
    GRUWithLinear(371, 128, num_layers=2, dropout=0.1, has_padding=False).cuda(),  # Smaller hidden
    GRUWithLinear(81, 64, num_layers=2, dropout=0.1, has_padding=False).cuda(),
    GRUWithLinear(300, 256, num_layers=2, dropout=0.1, has_padding=False).cuda()   # Text strong
]
head = MLP(512, 256, 2).cuda()  # TensorFusion out ~512 (author project) -> binary 2

fusion = TensorFusion().cuda()  # Dynamic fusion for classification

if __name__ == '__main__':
    train(encoders, fusion, head, traindata, validdata, 20, task="classification",  # Binary no regression
          optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-4,
          objective=simple_ce, save='sarcasm_tf_best.pt', weight_decay=0.01)

    print("Testing:")
    model = torch.load('sarcasm_tf_best.pt', weights_only=False).cuda()
    test(model=model, test_dataloaders_all=test_robust, dataset='sarcasm',
         is_packed=False, criterion=simple_ce, task='classification', no_robust=True, AUPRC=False)