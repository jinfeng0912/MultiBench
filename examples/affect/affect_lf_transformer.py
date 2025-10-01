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

# Load BERT data (sarcasm binary, text=768)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm_bert.pkl',
    robust_test=False, max_pad=True, task='classification')  # Binary sarcasm


# Ultra+ CE: robust for dim/mismatch (from GB fix)
def simple_ce(pred, target, *args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Pred check
    if pred is None or not isinstance(pred, torch.Tensor):
        return torch.tensor(0.0, requires_grad=True, device=device)

    if pred.numel() == 0 or pred.dim() == 0 or len(pred.shape) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    if pred.dim() > 2: pred = pred.mean(1)  # [B,seq,C] -> [B,C]
    pred = torch.clamp(pred, -10, 10)

    # Ensure dim=2 [B,C]
    if pred.dim() < 2:
        print("Warning: Pred dim <2 (skip)")
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Target
    if not isinstance(target, torch.Tensor) or target.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    target = target.long().squeeze(-1).to(pred.device)

    # Mismatch try
    try:
        p_b = pred.shape[0]
        t_b = target.shape[0]
        if p_b != t_b or p_b == 0:
            print(f"Warning: Mismatch B={p_b} vs {t_b} (skip)")
            return torch.tensor(0.0, requires_grad=True, device=device)
    except IndexError as e:
        print(f"Warning: Shape error {e} (skip)")
        return torch.tensor(0.0, requires_grad=True, device=device)

    return nn.CrossEntropyLoss()(pred, target)


# Sarcasm Transformer BERT dims (text input=768, hidden=600; concat 1100)
encoders = [
    Transformer(371, 400).cuda(),  # Visual
    Transformer(81, 100).cuda(),  # Audio
    Transformer(768, 600).cuda()  # Text BERT: input=768 (fix), hidden=600 strong
]
head = MLP(1100, 256, 2).cuda()  # 400+100+600=1100 pooled -> binary 2

fusion = Concat().cuda()  # Late [B,50,1100] or pooled

if __name__ == '__main__':
    freeze_support()

    train(encoders, fusion, head, traindata, validdata, 20, task="classification",
          optimtype=torch.optim.AdamW, early_stop=False, is_packed=False,
          lr=1e-4, objective=simple_ce,
          save='sarcasm_lf_transformer_best_bert.pt', weight_decay=0.01)

    print("Testing:")
    try:
        model = torch.load('sarcasm_lf_transformer_best_bert.pt', weights_only=False).cuda()
    except FileNotFoundError:
        print("Warning: No best model – test skipped")
        model = None
    if model:
        test(model=model, test_dataloaders_all=test_robust, dataset='sarcasm',
             is_packed=False, criterion=simple_ce,
             task='classification', no_robust=True, auprc=False)  # Fix: AUPRC -> auprc lowercase
    else:
        print("Test skipped")