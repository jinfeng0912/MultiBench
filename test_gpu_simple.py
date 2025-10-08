"""简单的GPU测试 - 验证GPU是否真正工作"""
import torch
import time

print("=" * 60)
print("GPU简单测试")
print("=" * 60)

# 1. 检查CUDA
print(f"\n1. CUDA可用: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("错误: CUDA不可用!")
    exit(1)

print(f"2. GPU: {torch.cuda.get_device_name(0)}")

# 2. 创建数据并移动到GPU
print("\n3. 创建张量并移动到GPU...")
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
print(f"   张量在GPU: {x.is_cuda}")
print(f"   显存使用: {torch.cuda.memory_allocated(0)/1024**2:.0f}MB")

# 3. 执行计算
print("\n4. 执行GPU计算...")
print("   提示: 现在用 nvidia-smi 应该能看到进程和GPU利用率!")

for i in range(100):
    z = torch.matmul(x, y)
    if i == 0:
        print(f"   第1次计算完成")
    if i == 50:
        print(f"   第50次计算完成")
        
torch.cuda.synchronize()
print(f"   100次计算完成!")

# 4. 更密集的计算
print("\n5. 执行密集计算（10秒）...")
print("   提示: nvidia-smi 的 GPU-Util 应该接近100%!")

start = time.time()
while time.time() - start < 10:
    z = torch.matmul(x, y)
    z = torch.matmul(z, x)
    
torch.cuda.synchronize()
print("   密集计算完成!")

print(f"\n6. 最终显存: {torch.cuda.memory_allocated(0)/1024**2:.0f}MB")
print("\n" + "=" * 60)
print("测试完成! 如果nvidia-smi显示了进程和GPU利用率,说明GPU工作正常")
print("=" * 60)
