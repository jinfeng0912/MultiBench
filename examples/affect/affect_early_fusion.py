import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
from multiprocessing import freeze_support


# 用我的 mosi_data.pkl (dims [20,5,300])
traindata, validdata, testdata = get_dataloader(
    r'E:\Laboratory\datasets\CMU_MOSI\mosi_bert.pkl',
    robust_test=False, max_pad=True, data_type='mosi')  # 移除 max_seq_len (您的 pkl 已 pad 到 50)

# mosi/mosei (匹配您的 dims: audio=20, visual=5, text=300; early concat=325)[35,74,768]
encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]  # Identity for raw feats
head = Sequential(
    GRU(877, 512, dropout=True, has_padding=False, batch_first=True, last_only=True),  # 877=35+74+768 (early concat)
    MLP(512, 512, 1)
).cuda()

# humor/sarcasm (注释掉，不用)
# encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
# head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True, last_only=True), MLP(1128, 512, 1)).cuda()

fusion = ConcatEarly().cuda()  # Early fusion: feats 先 concat

if __name__ == '__main__':
    freeze_support()

    print("Debug: Checking traindata batch shapes...")
    for batch in traindata:
        input_shapes = [x.shape for x in batch[:-1]]
        print("Train batch shapes (audio, visual, text):", input_shapes)  # 预期: [[32,20,50], [32,5,50], [32,300,50]]
        break

    train(encoders, fusion, head, traindata, validdata, 20,
          task="regression", optimtype=torch.optim.AdamW,
          is_packed=False, lr=1e-3, save='mosi_ef_r0_bert.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    print("Testing:")
    model = torch.load('mosi_ef_r0_bert.pt', weights_only=False).cuda()
    test(model=model, test_dataloaders_all=testdata, dataset='mosi', is_packed=False,
         criterion=torch.nn.L1Loss(), task="regression", no_robust=True)