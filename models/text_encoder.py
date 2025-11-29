from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc = nn.Linear(512, out_dim)

    def forward(self, text):
        # 如果text是一个字符串列表，我们需要处理批量输入
        if isinstance(text, list):
            # 对整个列表进行tokenize，支持不同的文本
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        else:
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
        # 将tokens移到与text_encoder相同的设备上
        device = next(self.text_encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        out = self.text_encoder(**tokens)
        pooled = out.pooler_output
        
        # 确保输出的批次大小与输入相同
        if isinstance(text, list):
            # 当输入是列表时，确保输出批次大小匹配
            # 模型会为每个文本生成独立的嵌入
            assert pooled.size(0) == len(text), f"Expected {len(text)} outputs, got {pooled.size(0)}"
            
        return self.fc(pooled)