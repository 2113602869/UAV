import torch
from torch.utils.data import DataLoader
from models.visual_backbone import VisualBackbone
from models.text_encoder import TextEncoder
from models.fusion_module import CrossModalFusion
from models.tracker import Tracker
from data.uav_dataset import UAV123Dataset
import yaml

def load_config():
    with open("configs/train.yaml", "r") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()

    dataset = UAV123Dataset(
        root=cfg["dataset"]["root"],
        split_json=cfg["dataset"]["split"],
        mode="train"
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    backbone = VisualBackbone()
    text = TextEncoder()
    fusion = CrossModalFusion()
    model = Tracker(backbone, text, fusion)

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 将模型移动到GPU（如果可用）
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.L1Loss()

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            img = batch["image"].to(device)
            box = batch["box"].to(device)

            pred = model(img, ["a drone target"] * img.size(0))  # 为每个样本复制文本提示
            loss = loss_fn(pred, box)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print("Epoch", epoch, "Loss", loss.item())

if __name__ == "__main__":
    train()