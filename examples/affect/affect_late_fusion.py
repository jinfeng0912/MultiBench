import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 单 GPU

from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch
from multiprocessing import freeze_support  # Windows 兼容


# 用您的 mosi_data.pkl (dims [20,5,300])
traindata, validdata, test_robust = get_dataloader(
    r'E:\Laboratory\datasets\CMU_MOSI\mosi_bert.pkl',
    robust_test=False, max_pad=True)  # 移除 data_type (默认 mosi)，您的 pkl 已 pad

# mosi/mosei (匹配您的 dims: hidden [40,20,600] sum=660 for late concat)
# 加 last_only=True: GRU 输出 [B, H] (last timestep), 非 [B,T,H]
encoders = [
    GRU(35, 100, dropout=True, has_padding=False, batch_first=True, last_only=True).cuda(),  # visual: [B,50,35] -> [B,100]
    GRU(74, 150, dropout=True, has_padding=False, batch_first=True, last_only=True).cuda(),  # audio: [B,50,74] -> [B,150]
    GRU(768, 512, dropout=True, has_padding=False, batch_first=True, last_only=True).cuda()  # text: [B,50,768] -> [B,512]
]
head = MLP(762, 762, 1).cuda()

# 标准 dims 注释 (不用)
# encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True, last_only=True).cuda(),
#             GRU(74, 200, dropout=True, has_padding=True, batch_first=True, last_only=True).cuda(),
#             GRU(300, 600, dropout=True, has_padding=True, batch_first=True, last_only=True).cuda()]
# head = MLP(870, 870, 1).cuda()

# humor/sarcasm (注释掉，不用)
# ...

fusion = Concat().cuda()  # Late: 独立 [B,H] concat -> [B,660]

if __name__ == '__main__':
    freeze_support()  # Windows 兼容

    # Debug: Print batch shapes
    print("Debug: Checking traindata batch shapes...")
    for batch in traindata:
        input_shapes = [x.shape for x in batch[:-1]]
        print("Train batch shapes (audio, visual, text):", input_shapes)
        break

    train(encoders, fusion, head, traindata, validdata, 20,  # 20 epochs
          task="regression", optimtype=torch.optim.AdamW,
          early_stop=False, is_packed=False, lr=1e-3, save='mosi_lf_best_bert.pt',
          weight_decay=0.01, objective=torch.nn.L1Loss(), clip_val=1.0)  # 加 clip_val 如 MULT

    print("Testing:")
    model = torch.load('mosi_lf_best_bert.pt', weights_only=False).cuda()
    test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
         criterion=torch.nn.L1Loss(), task='regression', no_robust=True)