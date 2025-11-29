import torch

# 检查 CUDA 是否可用
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 查看可用的 GPU 数量
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    # 查看当前 GPU 的名称
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")