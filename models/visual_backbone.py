import torch.nn as nn
import torch

class VisualBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 我们将在第一次前向传播时动态确定全连接层的输入维度
        self.fc = None
        self.out_dim = out_dim
        self._fc_initialized = False

    def forward(self, x):
        feat = self.net(x)
        batch_size = feat.size(0)
        
        # 如果这是第一次前向传播，初始化全连接层
        if not self._fc_initialized:
            # 计算展平后的特征维度
            flattened_dim = feat.size(1) * feat.size(2) * feat.size(3)
            self.fc = nn.Linear(flattened_dim, self.out_dim)
            # 将模块移动到相同的设备上
            device = next(self.net.parameters()).device
            self.fc = self.fc.to(device)
            self._fc_initialized = True
            
        feat = feat.view(batch_size, -1)
        feat = self.fc(feat)
        return feat