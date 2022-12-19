import torch
from torch.utils.data import dataloader
from typing import List, Union


class Trainer:
    """
    세그멘테이션 모델 학습을 위한 클래스
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, metric_fn: Union[torch.nn.Module, torch.nn.Module],
                 optimizer: torch.optim.Optimizer, device: str, len_epoch: int,
                 data_loader: torch.utils.data.dataloader, valid_data_loader: torch.utils.data.dataloader = None,
                 lr_scheduler: torch.optim.lr_scheduler = None):

        # CUDA // device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # make_dataloder 함수의 결과 값
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        # config.json 파일로부터 파라미터를 호출받고, getattr로 생성된 객체
        self.lr_scheduler = lr_scheduler

        # model/metric.py의 함수들 전부 호출
        self.metric_fn = metric_fn

        # model/model.py -> U-Net, DeepLab v3, FCN, etc..
        self.model = model

        # config.json 파일로부터 파라미터를 호출받고, getattr로 생성된 객체
        self.criterion = criterion
        self.optimizer = optimizer

        # config.json 파일로부터 파라미터를 호출받아 생성된 int 변수
        self.epochs = len_epoch

        self.log = dict()
        self.log['train_loss'] = []
        self.log['val_loss'] = []
        self.log['train_metric'] = []
        self.log['val_metric'] = []

    def _train_epoch(self, epoch: int):
        train_loss, train_metric = 0, 0

        self.model.train()
        for batch, (x_train, y_train) in enumerate(self.data_loader):
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            y_pred = self.model(x_train)
            loss = self.criterion(y_pred, y_train)

            train_loss += loss.item()
            train_metric += self.metric_fn(y_pred, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.data_loader)
        train_metric /= len(self.data_loader)
        print(f'Train Loss : {train_loss:.5f} | Train Metric : {train_metric:.2f}% | ')
        self.log['train_loss'].append(train_loss)
        self.log['train_metric'].append(train_metric)

        if self.do_validation:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_epoch(self, epoch: int):
        val_metric, val_loss = 0, 0

        self.model.eval()
        with torch.inference_mode():
            for (x_test, y_test) in self.valid_data_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                y_pred = self.model(x_test)
                loss = self.criterion(y_pred, y_test)

                val_loss += loss.item()
                val_metric += self.metric_fn(y_pred, y_test)

            val_loss /= len(self.valid_data_loader)
            val_metric /= len(self.valid_data_loader)
            print(f'Val Loss : {val_loss:.5f} | Val Metric : {val_metric:.2f}%')
            self.log['val_loss'].append(val_loss)
            self.log['val_metric'].append(val_metric)

    def train(self):
        for epoch in range(self.epochs):
            print(f'\nEpoch : {epoch} | ')
            self._train_epoch(epoch)

        return self.log
