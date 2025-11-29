import torch.nn as nn
import torch

class Tracker(nn.Module):
    def __init__(self, backbone, text_encoder, fusion):
        super().__init__()
        self.backbone = backbone
        self.text = text_encoder
        self.fusion = fusion
        self.box_head = nn.Linear(256, 4)

    def forward(self, img, text):
        v = self.backbone(img)
        t = self.text(text)
        fused = self.fusion(v, t)
        pred_box = self.box_head(fused)
        return pred_box