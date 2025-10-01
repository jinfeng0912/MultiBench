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
from fusions.common_fusions import Concat  # noqa (stable for BERT)

warnings.filterwarnings('ignore', category=UserWarning)

# Load BERT data (sarcasm binary classification, text=768 seq=50)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm_bert.pkl',
    robust_test=False, max_pad=True, task='classification')


# Ultra+ CE: robust (post-head pred [B,2], no need but safe)
def simple_ce(pred, target, *args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if pred is None or not isinstance(pred, torch.Tensor):
        return torch.tensor(0.0, requires_grad=True, device=device)

    if pred.numel() == 0 or pred.dim() == 0 or len(pred.shape) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    if pred.dim() > 2: pred = pred.mean(1)  # Rarely [B,seq,2] if no flatten
    pred = torch.clamp(pred, -10, 10)

    if pred.dim() < 2:
        print("Warning: Pred dim <2 (skip)")
        return torch.tensor(0.0, requires_grad=True, device=device)

    if not isinstance(target, torch.Tensor) or target.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    target = target.long().squeeze(-1).to(pred.device)

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


# Encoders BERT (out [B,50,h]; concat_dim=448)
encoders = [
    GRUWithLinear(371, 128, 128, dropout=0.1, has_padding=False).cuda(),  # [B,50,128]
    GRUWithLinear(81, 64, 64, dropout=0.1, has_padding=False).cuda(),  # [B,50,64]
    GRUWithLinear(768, 256, 256, dropout=0.1, has_padding=False).cuda()  # [B,50,256]
]

# Head: Flatten [B,50,448] -> [B,50*448=22400] in MLP + Linear(22400,256)
head = MLP(22400, 256, 2).cuda()  # Match flatten (seq=50 * concat=448); out [B,256] -> [B,2]

fusion = Concat().cuda()  # [B,50,448]

if __name__ == '__main__':
    # train(encoders, fusion, head, traindata, validdata, 20, task="classification",
    #       optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-4,
    #       objective=simple_ce, save='sarcasm_tf_best_bert.pt', weight_decay=0.01)

    print("Testing:")
    model_path = 'sarcasm_tf_best_bert.pt'
    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False).cuda()
        test(model=model, test_dataloaders_all=test_robust, dataset='sarcasm',
             is_packed=False, criterion=simple_ce, task='classification', no_robust=True,
             auprc=False)  # Fix: AUPRC -> auprc lowercase
    else:
        print("No model – train first")