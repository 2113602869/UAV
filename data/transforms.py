import cv2
import numpy as np

def default_transforms(image, bbox=None, size=(256, 256)):
    """
    对图像和边界框进行预处理
    :param image: 输入图像 (HWC, BGR格式)
    :param bbox: 边界框 (x1, y1, x2, y2)，可选
    :param size: 目标尺寸 (width, height)
    :return: 处理后的图像和边界框
    """
    # 1. 调整图像尺寸
    h, w = image.shape[:2]
    scale_w = size[0] / w
    scale_h = size[1] / h
    image = cv2.resize(image, size)

    # 2. 调整边界框（如果有）
    if bbox is not None:
        bbox = np.array(bbox, dtype=np.float32)
        bbox[0] *= scale_w  # x1
        bbox[1] *= scale_h  # y1
        bbox[2] *= scale_w  # x2
        bbox[3] *= scale_h  # y2

    # 3. BGR → RGB（如果模型需要RGB输入）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. 维度转换：HWC → CHW（PyTorch模型通常需要CHW格式）
    image = image.transpose(2, 0, 1)

    # 5. 归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0

    return image, bbox

def inverse_transform(bbox, original_size, target_size=(256, 256)):
    """
    将处理后的边界框转换回原始图像尺寸
    :param bbox: 处理后的边界框 (x1, y1, x2, y2)
    :param original_size: 原始图像尺寸 (width, height)
    :param target_size: 预处理时的目标尺寸 (width, height)
    :return: 原始尺寸的边界框
    """
    if bbox is None:
        return None
    scale_w = original_size[0] / target_size[0]
    scale_h = original_size[1] / target_size[1]
    bbox = np.array(bbox, dtype=np.float32)
    bbox[0] *= scale_w
    bbox[1] *= scale_h
    bbox[2] *= scale_w
    bbox[3] *= scale_h
    return bbox