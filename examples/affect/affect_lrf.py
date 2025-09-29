import torch
import sys
import os
import torch.nn as nn  # for Linear, Sequential, Module
import torch.nn.functional as F  # for norm math

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 单 GPU

from private_test_scripts.all_in_one import all_in_one_train  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from unimodals.common_models import GRU, MLP, Sequential  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import LowRankTensorFusion  # noqa
from multiprocessing import freeze_support  # Windows 兼容


# 用您的 mosi_data.pkl (dims [20,5,300])
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\CMU_MOSI\mosi_bert.pkl',
    robust_test=False, max_pad=True
)


# 自定义 Normalize Module (继承 nn.Module，支持 Sequential)
class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # z-score norm over feat dim (T axis kept; per sample/seq)
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)


# mosi/mosei: Sequential(Norm + GRU last_only=True + Linear) → 静态 [B, linear_dim]
# dims: audio [20→40→32], visual [5→20→32], text [300→600→128]
encoders = [
    Sequential(  # Visual
        # Normalize(),  # 注释: BERT feats often pre-normed; enable if need
        GRU(35, 80, dropout=0.1, has_padding=False, batch_first=True, last_only=True),
        nn.Linear(80, 32)
    ).cuda(),
    Sequential(  # Audio
        # Normalize(),
        GRU(74, 100, dropout=0.1, has_padding=False, batch_first=True, last_only=True),
        nn.Linear(100, 32)
    ).cuda(),
    Sequential(  # Text
        # Normalize(),
        GRU(768, 300, dropout=0.1, has_padding=False, batch_first=True, last_only=True),
        nn.Linear(300, 128)
    ).cuda()
]
head = MLP(128, 512, 1).cuda()

fusion = LowRankTensorFusion([32, 32, 128], 128, 32).cuda()  # 输入 [32,32,128], 输出 128, rank=32

if __name__ == '__main__':
    freeze_support()

    # Debug: batch shapes
    print("Debug: Checking traindata batch shapes...")
    for batch in traindata:
        input_shapes = [x.shape for x in batch[:-1]]
        print("Train batch shapes (audio, visual, text):", input_shapes)
        break

    # 调参: lr低, epochs多, early_stop False, clip小, dropout已减
    train(encoders, fusion, head, traindata, validdata, 30,  # 30 epochs
          task="regression", optimtype=torch.optim.AdamW,
          early_stop=False, is_packed=False, lr=1e-4,  # lr=1e-4 关键
          save='mosi_lrf_v4_bert.pt', weight_decay=0.01,
          objective=torch.nn.L1Loss(), clip_val=0.5)  # clip小

    print("Testing:")
    model = torch.load('mosi_lrf_v4_bert.pt', weights_only=False).cuda()
    test(model, test_robust, dataset='mosi', is_packed=False,
         criterion=torch.nn.L1Loss(), task='regression', no_robust=True)