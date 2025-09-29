import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from fusions.mult import MULTModel
from unimodals.common_models import Identity, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from multiprocessing import freeze_support


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\CMU_MOSI\mosi_bert.pkl',
    robust_test=False,
    max_pad=True)


class HParams():
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0,0,0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 1
        all_steps = False

if __name__ == '__main__':
    freeze_support()  # Added for Windows; safe to include

    # Debug: Print actual input shapes from dataloader (audio, visual, text)
    print("Debug: Checking traindata batch shapes...")
    for batch in traindata:
        input_shapes = [x.shape for x in batch[:-1]]  # Exclude labels
        print("Train batch shapes (audio, visual, text):", input_shapes)
        # Expected: [batch, feat_dim, seq_len], e.g., [[32, 20, 50], [32, 5, 50], [32, 300, 50]]
        break  # Only check first batch

    encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
    fusion = MULTModel(3, [35,74,768], HParams()).cuda()  # Fixed: Adjusted hidden_dims to match your custom data (audio=20, visual=5, text=300)
    # If debug shows different dims, update here, e.g., [20, 35, 300] for standard visual
    # Alternative for standard MOSI: fusion = MULTModel(3, [74, 35, 300], HParams()).cuda()
    head = Identity().cuda()

    train(encoders, fusion, head, traindata, validdata, 20, task="regression",
          optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-3,
          clip_val=1.0, save='mosi_mult_best_bert.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    print("Testing:")
    model = torch.load('mosi_mult_best_bert.pt', weights_only=False).cuda()

    test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
         criterion=torch.nn.L1Loss(), task='regression', no_robust=True)  # Fixed: Changed task to 'regression' to match training (L1Loss for sentiment regression on MOSI)