import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

# 显示GPU状态
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# 调整参数适配WSL环境
print("正在加载数据集...")
traindata, validdata, testdata = get_dataloader(
    '/home/hejinfeng/datasets/AV_MNIST',
    batch_size=16,
    num_workers=0)
print("数据集加载完成！")
print("=" * 50)

channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

fusion = Concat().cuda()

# 显示模型和显存状态
if torch.cuda.is_available():
    print(f"模型已加载，显存使用: {torch.cuda.memory_allocated(0)/1024**2:.0f}MB")
print("开始训练...")
print("=" * 50)

# 训练时在另一个终端运行 nvidia-smi 监控GPU使用
train(encoders, fusion, head, traindata, validdata, 1,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001,
      save='avmnist_simple_late_fusion_best.pt')

print("Testing:")
model = torch.load('avmnist_simple_late_fusion_best.pt').cuda()
test(model, testdata, no_robust=True)
