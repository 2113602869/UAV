import torch.nn as nn
import torch

class CrossModalFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, v_feat, t_feat):
        # 确保文本特征与视觉特征具有相同的批次大小
        if v_feat.size(0) != t_feat.size(0):
            if t_feat.size(0) == 1:
                # 扩展文本特征以匹配视觉特征的批次大小
                t_feat = t_feat.expand(v_feat.size(0), -1)
            else:
                # 其他情况，截取或适配张量
                raise ValueError(f"无法对齐视觉特征({v_feat.shape})和文本特征({t_feat.shape})的批次大小")
                
        fused = torch.cat([v_feat, t_feat], dim=-1)
        fused = self.fc(fused)
        return v_feat + self.gate * fused