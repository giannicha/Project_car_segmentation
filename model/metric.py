import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def metric(loader: DataLoader, model: nn.Module, device: str = device_):
    """
    세그멘테이션 모델 평가 함수

    :param loader: 평가 데이터셋
    :param model: 세그멘테이션 모델
    :param device: GPU 설정
    :return: Pixel accuracy, Dice score
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(torch.softmax(model(x), dim=1), dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    pixel_acc = (num_correct / num_pixels) * 100
    dice_score = dice_score / len(loader)
    model.train()
    return {'pixel_acc' : pixel_acc, 'dice_score' : dice_score}