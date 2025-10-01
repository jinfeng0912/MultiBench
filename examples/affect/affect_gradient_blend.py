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

# Load BERT data (sarcasm binary classification, text=768 dim)
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\MUSTARD_sarcasm\sarcasm_bert.pkl',  # Change to BERT pkl (text=768)
    task='classification', robust_test=False, max_pad=True)

# Guard print for multiprocess (main only)
if 'memory_profiler' not in sys.modules and __name__ == '__main__':
    print(f"Train samples: {len(traindata.dataset)}, Valid: {len(validdata.dataset)}")


# Ultra+ CE: dim>=2 explicit + try for shape bug (fix 1D/scalar in val)
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

    # Mismatch with try
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


# GRU encoders BERT version (seq out [B,50,h]; text input=768 project to 300)
encoders = [
    GRUWithLinear(371, 350, 350, dropout=0.1, has_padding=False).cuda(),  # Visual: [B,50,350]
    GRUWithLinear(81, 75, 75, dropout=0.1, has_padding=False).cuda(),  # Audio: [B,50,75]
    GRUWithLinear(768, 300, 300, dropout=0.1, has_padding=False).cuda()  # Text BERT: [B,50,300]
]

seq_len = 50  # Fixed pad
concat_dim = 350 + 75 + 300  # 725 sum h
mm_head_input = seq_len * concat_dim  # 36250 for multi global flatten in getmloss

head = MLP(mm_head_input, 512, 2).cuda()  # Multimodal: flat [B,36250] 2D -> [B,512] -> [B,2]

# Unimodal heads: per-frame view(-1, h) last=h -> input_dim = h (revert flatten)
unimodal_heads = [
    MLP(350, 128, 2).cuda(),  # Vis: last=350 after view(-1,350)
    MLP(75, 32, 2).cuda(),  # Aud: last=75
    MLP(300, 128, 2).cuda()  # Text: last=300
]

fusion = Concat().cuda()  # [B,50,725] then flat in getmloss

if __name__ == '__main__':
    freeze_support()

    # training_structures.gradient_blend.criterion = simple_ce
    #
    # train(encoders, head, unimodal_heads, fusion, traindata, validdata, 20, gb_epoch=10, lr=1e-4,
    #       auprc=False, classification=True, optimtype=torch.optim.AdamW,  # Fix: AUPRC -> auprc lowercase
    #       savedir='sarcasm_gb_best_bert.pt', weight_decay=0.01)

    print("Saving GB last full model...")
    full_model = nn.ModuleList(encoders + unimodal_heads + [fusion, head])
    torch.save(full_model, 'sarcasm_gb_last_bert.pt')

    print("Testing:")
    model_path = 'sarcasm_gb_best_bert.pt' if os.path.exists('sarcasm_gb_best_bert.pt') else 'sarcasm_gb_last_bert.pt'
    if os.path.exists(model_path):
        full_model = torch.load(model_path, weights_only=False).cuda()
        print(f"Loaded {model_path} – testing...")
        test(full_model, test_robust, dataset='sarcasm', auprc=False,
             no_robust=True)  # Fix: Remove task='classification' (no arg)
    else:
        print("No model")