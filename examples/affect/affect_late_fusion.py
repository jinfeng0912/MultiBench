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
    r'E:\Laboratory\datasets\CMU_MOSI\mosi_data.pkl',
    robust_test=False, max_pad=True, data_type='mosi')  # 移除原 raw.pkl 和 sarcasm 注释

# mosi/mosei (匹配您的 dims: hidden [40,20,600] sum=660 for late concat)
encoders = [
    GRU(20, 40, dropout=True, has_padding=False, batch_first=True).cuda(),  # audio=20 -> hidden 40
    GRU(5, 20, dropout=True, has_padding=False, batch_first=True).cuda(),   # visual=5 -> 20
    GRU(300, 600, dropout=True, has_padding=False, batch_first=True).cuda() # text=300 -> 600
]
head = MLP(660, 660, 1).cuda()  # 660=40+20+600 (late concat 后)

# 标准 dims 注释 (不用)
# encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
#             GRU(74, 200, dropout=True, has_padding=True, batch_first=True).cuda(),
#             GRU(300, 600, dropout=True, has_padding=True, batch_first=True).cuda()]
# head = MLP(870, 870, 1).cuda()

# humor/sarcasm (注释掉，不用)
# encoders=[GRU(371,512,dropout=True,has_padding=False, batch_first=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=False, batch_first=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=False, batch_first=True).cuda()]
# head=MLP(1368,512,1).cuda()

fusion = Concat().cuda()  # Late fusion: 独立编码后 concat

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
          early_stop=False, is_packed=False, lr=1e-3, save='mosi_lf_best.pt',
          weight_decay=0.01, objective=torch.nn.L1Loss())  # is_packed=False 匹配您的设置

    print("Testing:")
    model = torch.load('mosi_lf_best.pt', weights_only=False).cuda()
    test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
         criterion=torch.nn.L1Loss(), task='regression', no_robust=True)  # 统一 regression