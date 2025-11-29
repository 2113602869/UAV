import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from data.transforms import default_transforms

class UAV123Dataset(Dataset):
    def __init__(self, root, split_json, mode="train", transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        with open(split_json, "r") as f:
            split = json.load(f)

        self.seqs = split[mode]

        self.samples = []
        for seq in self.seqs:
            # 修正图像目录路径
            img_dir = os.path.join(root, "data_seq", "UAV123", seq)
            # 修正ground truth文件路径
            gt_path = os.path.join(root, "anno", "UAV123", seq + ".txt")

            # 检查文件是否存在
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth file {gt_path} does not exist, skipping sequence {seq}")
                continue

            with open(gt_path, "r") as f:
                lines = f.readlines()

            for i, box in enumerate(lines):
                img_path = os.path.join(img_dir, f"{i+1:06d}.jpg")
                # 检查图像文件是否存在
                if not os.path.exists(img_path):
                    continue
                    
                x, y, w, h = map(float, box.strip().split(","))
                # 从 (x, y, w, h) 转换为 (x1, y1, x2, y2)
                x1, y1, x2, y2 = x, y, x+w, y+h
                self.samples.append((img_path, [x1, y1, x2, y2]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box = self.samples[idx]
        img = cv2.imread(img_path)
        
        if self.transform:
            img, box = self.transform(img, box)
        else:
            img, box = default_transforms(img, box)

        return {
            "image": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
        }