from torch import nn
import torch
import sys
import os
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train  # noqa
import training_structures  # noqa
from training_structures.gradient_blend import train, test  # noqa
from unimodals.common_models import GRUWithLinear, MLP  # GRU fallback
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import Concat  # noqa
from multiprocessing import freeze_support

# Suppress warnings (author clean output)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules')

# Load data (sarcasm binary classification)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm.pkl',
    task='classification', robust_test=False, max_pad=True)

# Simple CE: pool seq + clamp (author minimal stable loss)
def simple_ce(pred, target, *args):
    if pred.dim() > 2: pred = pred.mean(1)  # [B,seq,C] -> [B,C]
    pred = torch.clamp(pred, -10, 10)
    target = target.long().squeeze(-1).to(pred.device)
    return nn.CrossEntropyLoss()(pred, target)

# GRU encoders (stable, no NaN; dims for concat 1450: out 700/150/600)
encoders = [
    GRUWithLinear(371, 350, num_layers=2, has_padding=False, dropout=0.1).cuda(),  # Out ~700 (pooled)
    GRUWithLinear(81, 75, num_layers=2, has_padding=False, dropout=0.1).cuda(),   # Out ~150
    GRUWithLinear(300, 300, num_layers=2, has_padding=False, dropout=0.1).cuda()  # Out ~600
]
head = MLP(1450, 512, 2).cuda()  # Concat 1450 -> binary 2

unimodal_heads = [
    MLP(700, 512, 2).cuda(),  # Visual uni
    MLP(150, 64, 2).cuda(),   # Audio uni
    MLP(600, 256, 2).cuda()   # Text uni
]

fusion = Concat().cuda()

if __name__ == '__main__':
    freeze_support()

    # Set simple criterion (author internal loss)
    training_structures.gradient_blend.criterion = simple_ce

    train(encoders, head, unimodal_heads, fusion, traindata, validdata, 20, gb_epoch=10, lr=1e-4,
          AUPRC=False, classification=True, optimtype=torch.optim.AdamW,
          savedir='sarcasm_gb_best.pt', weight_decay=0.01)

    # Manual last save full (match author completeModule)
    print("Saving GB last full model...")
    full_model = nn.ModuleList(encoders + unimodal_heads + [fusion, head])
    torch.save(full_model, 'sarcasm_gb_last.pt')

    print("Testing:")
    model_path = 'sarcasm_gb_best.pt' if os.path.exists('sarcasm_gb_best.pt') else 'sarcasm_gb_last.pt'
    if os.path.exists(model_path):
        full_model = torch.load(model_path, weights_only=False).cuda()
        print(f"Loaded {model_path} – testing...")
        test(full_model, test_robust, dataset='sarcasm', auprc=False, no_robust=True, task='classification')
    else:
        print("No model")