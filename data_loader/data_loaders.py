import torch
import glob
import numpy as np
import albumentations as A

from PIL import Image
from typing import Union, Tuple, List
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2


class CustomDataset(torch.utils.data.Dataset):
    """
    커스텀 데이터셋 생성을 위한 클래스

    :param img_path: 이미지 데이터가 포함된 리스트
    :param mask_path: 라벨 데이터가 포함된 리스트
    :param train: 학습 데이터 : True / 평가 데이터 : False
    :param transform: 전처리 객체
    """
    def __init__(self, img_path: List, mask_path: List, img_size: Tuple = (224, 224),
                 train: bool = True, transform: transforms.Compose = None):

        # 이미지 경로 설정
        self.img_data = img_path
        self.mask_data = mask_path

        # 데이터 폴더 개수 및 학습 데이터 비율 설정
        folder_num = 5
        split_ratio = 0.8

        # 학습 데이터 인덱스 = (전체 이미지 개수 * 학습 데이터 비율) / 폴더 개수
        train_idx = int(len(self.img_data) * split_ratio / folder_num)

        # 전처리
        self.transform = transform

        if train:
            # Case 1. 학습 데이터
            # 각 폴더별로 1000개씩 데이터가 존재하는 것을 확인. -> train_idx = 800
            # 각 폴더로부터 학습 데이터 인덱스만큼 데이터를 가져온다.
            self.img_data = sum(
                [[self.img_data[idx_1 + 1000 * idx_2] for idx_1 in range(train_idx)] for idx_2 in range(folder_num)], [])
            self.mask_data = sum(
                [[self.mask_data[idx_1 + 1000 * idx_2] for idx_1 in range(train_idx)] for idx_2 in range(folder_num)], [])
        else:
            # Case 2. 평가 데이터
            # 각 폴더로부터 학습 데이터 인덱스 이후로 모든 데이터를 가져온다.
            self.img_data = sum(
                [[self.img_data[idx_1 + 1000 * idx_2] for idx_1 in range(train_idx, int(len(self.img_data) / folder_num))]
                 for idx_2 in range(folder_num)], [])
            self.mask_data = sum(
                [[self.mask_data[idx_1 + 1000 * idx_2] for idx_1 in range(train_idx, int(len(self.mask_data) / folder_num))]
                 for idx_2 in range(folder_num)], [])

    def __len__(self):
        """
        데이터셋의 크기를 반환하는 함수

        :return len(self.img_data)
        """
        return len(self.img_data)

    def __getitem__(self, idx):
        """
        데이터셋의 샘플을 인덱스로 반환하는 함수

        :param idx: 인덱스
        :return: img, label
        """
        img = np.array(Image.open(self.img_data[idx]))
        label = np.array(Image.open(self.mask_data[idx]))

        aug = self.transform(image=img, mask=label)
        img = aug['image']
        label = aug['mask']
        label = torch.max(label, dim=2)[0]

        return img, label


if __name__ == '__main__':
    # Config.json에 들어가야 할 것들 : img_size, mean, std
    # train.py에 들어가야 할 것들 : img_path, mask_path, train_transform, CustomDataset

    img_path = sum([glob.glob(f'../Data/data{i}/data{i}/CameraRGB/*') for i in ['A', 'B', 'C', 'D', 'E']], [])
    mask_path = sum([glob.glob(f'../Data/data{i}/data{i}/CameraSeg/*') for i in ['A', 'B', 'C', 'D', 'E']], [])
    img_size = (160, 240)
    transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_data = CustomDataset(img_path=img_path, mask_path=mask_path, img_size=img_size, train=True,
                               transform=transform)
    test_data = CustomDataset(img_path=img_path, mask_path=mask_path, img_size=img_size, train=False,
                              transform=transform)
