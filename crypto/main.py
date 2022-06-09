from __future__ import annotations

import asyncio
import math
import warnings
from abc import abstractmethod
from datetime import datetime
# from binance import AsyncClient
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard import program
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
# from torch.utils.py.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler

# from typing import Self
from crypto.clients import Client, BinanceMarketAgent
from crypto.converters import DataFrameSampleConverter
from crypto.data import Token, Sample, Market, Samples
from crypto.modules import LinearEx, BatchNorm1dEx
from crypto.utils import Cache, PerformanceTimer, register_signals

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Config:
    tokens: [str] = [
        "BUSD",
        "USDT",
        "USDC",
        "BTC",
        "ETH",
        "BNB",
    ]
    markets: [(str, str)] = [
        ("BTC", "USDT"),
        # ("BTC", "BUSD"),
        # ("ETH", "USDT"),
        # ("ETH", "BUSD"),
        # ("BNB", "USDT"),
        # ("BNB", "BUSD"),
    ]
    seed: int = None  # 880712
    network_path: str = "data/network.pt"


class Crypto:
    tokens: dict[str, Token] = {}
    markets: [] = []
    _binanceMarketAgent = BinanceMarketAgent()

    def __init__(self, config: Config):
        for token_id, symbol in enumerate(config.tokens):
            self.tokens[symbol] = Token(token_id, symbol)
        for token1, token2 in config.markets:
            market = Market(self._binanceMarketAgent, self.get_token(token1), self.get_token(token2))
            self.markets.append(market)

    def get_token(self, symbol: str) -> Token:
        if symbol not in self.tokens:
            raise IndexError("Unknown token symbol")
        return self.tokens[symbol]


class SampleAnalyzer:
    def __init__(self, samples: Samples):
        self.samples = samples

    def get_in_shape(self):
        # train_feature = self.get_feature(self.data_frames[0])
        # print(self.samples[0].feature)
        in_shape = np.asarray(self.samples[0].feature).shape
        print("estimated in shape: ", in_shape)
        return in_shape

    def get_out_size(self):
        out_size = max(self.samples.labels) + 1
        print("estimated out nodes: ", out_size)
        return out_size

    def get_weights(self):
        # targets = np.array([self.get_label(x) for x in self.data_frames])
        occurrences = np.bincount(self.samples.labels)
        # occurrences.resize(5)
        weights = len(self.samples.labels) / (len(occurrences) * occurrences)
        return weights


class MarketDataset(Dataset):
    def __init__(self, samples: Samples, device: torch.device = None):
        self.samples = samples
        self.device = device
        self.features = torch.as_tensor(np.asarray(samples.features),
                                        device=device,
                                        dtype=torch.float)
        self.labels = torch.as_tensor(np.asarray(samples.labels),
                                      device=device,
                                      dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class FCNetwork(nn.Module):
    def __init__(self, in_shape: tuple, out_size: int):
        super(FCNetwork, self).__init__()
        hidden_size = 256
        # hidden_size = math.floor(math.pow(2, math.floor(math.log2(max(input_size, out_size)))))
        self.body_layers = nn.Sequential(
            nn.Linear(np.prod(in_shape), hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        return self.body_layers(x.flatten(1)), h


class Network(nn.Module):
    def __init__(self, in_shape: tuple, out_size: int):
        super(Network, self).__init__()
        seq_len, input_size = in_shape
        hidden_size = 64
        # hidden_size = math.floor(math.pow(2, math.floor(math.log2(max(input_size, out_size)))))
        self.body_layers = nn.Sequential(
            LinearEx(input_size, hidden_size),
            nn.ReLU(),
            BatchNorm1dEx(hidden_size)
        )
        self.gru_layers = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)

        self.head = nn.Linear(hidden_size, out_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        # (N, L, C)
        x = self.body_layers(x)
        x, h = self.gru_layers(x, h)

        if x.dim() == 3:
            # get last sequence values
            # (N, L, C) => (N, C)
            x = x[:, -1, :]

        x = self.head(x)
        return x, h


class MarketAI:
    _writer: SummaryWriter = None
    _network: nn.Module = None
    _optimizer: Optimizer = None
    _schedular: any = None
    _epoch = 0
    _latest_loss = 0
    _best_loss = float("inf")
    _train_dataloader = None
    _eval_dataloader = None
    _klines: pd.DataFrame = None
    _converter: DataFrameSampleConverter = None
    _train_frames: pd.DataFrame = None
    _eval_frames: pd.DataFrame = None

    def __init__(self, crypto: Crypto, config: Config):
        self._crypto = crypto
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = Cache(self._config.network_path,
                            load_fn=lambda f: torch.load(f, map_location=self._device),
                            save_fn=lambda d, f: torch.save(f, d))

    def init(self):
        self.check()
        self.launch_tensor_board()

    def create_network(self, in_nodes, out_nodes):
        if self._config.seed is not None:
            torch.manual_seed(self._config.seed)
        self._network = Network(in_nodes, out_nodes).to(self._device)
        self._optimizer = optim.AdamW(self._network.parameters(), lr=1e-4)
        # optimizer = optim.SGD(network.parameters(), lr=1e-4, momentum=0.9)
        # schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self._schedular = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, verbose=True)
        self._epoch = 0

    def load_network(self):
        checkpoint = self._cache.load()
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        latest_loss = checkpoint["loss"]
        self._best_loss = latest_loss
        print(f"Network loaded with {epoch} trained, latest loss {latest_loss:.4f}")

    def save_network(self):
        checkpoint = {
            "epoch": self._epoch,
            "loss": self._latest_loss,
            "model_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict()
        }
        self._cache.update(checkpoint)

    # def get_samples(self, threshold: float = 0.8) -> (Samples, Samples):
    #     samples = Samples.from_data_frames(self._data_frames)
    #     split_index = math.floor(len(samples) * threshold)
    #     training_samples = samples[:split_index]
    #     eval_samples = samples[split_index:]
    #     return training_samples, eval_samples

    @staticmethod
    def check():
        if torch.cuda.is_available():
            print(f"CUDA {torch.version.cuda} (Devices: {torch.cuda.device_count()})")
        if torch.backends.cudnn.enabled:
            # torch.backends.cudnn.benchmark = True
            print(f"CUDNN {torch.backends.cudnn.version()}")

    def launch_tensor_board(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', "runs"])
        url = tb.launch()
        print("tensorboard url:", url)
        self._writer = SummaryWriter()

    # def get_weights(self, dataset):
    #     # targets = np.array([self.get_label(x) for x in self.data_frames])
    #     occurrences = dataset.labels.to(self._device).bincount()
    #     # occurrences.resize(5)
    #     weights = len(dataset.labels) / (len(occurrences) * occurrences)
    #     return weights

    def run(self):

        market = self._crypto.markets[0]
        self._klines = market.klines
        split_index = math.floor(len(self._klines) * 0.8)
        self._train_frames = self._klines[:split_index]
        self._eval_frames = self._klines[split_index:]
        self._converter = DataFrameSampleConverter(self._klines)
        # init network
        try:
            self.load_network()
        except Exception as e:
            print("Failed to load the network.")
            sample = self._converter.get_sample(self._klines[0])
            self.create_network(Helper.get_in_shape(sample), 3)

        print("Creating Trainer")
        trainer = Trainer(self._train_frames, self._converter, self._network, self._optimizer, self._device)
        print("Creating Evaluator")
        evaluator = Evaluator(self._eval_frames, self._converter, self._network, self._device)
        print("Creating ROIEvaluator")
        roi_evaluator = ROIEvaluator(self._eval_frames, self._converter, self._network, self._device)

        while True:
            # print("lr", optimizer.param_groups[0]['lr'])
            self._epoch += 1
            epoch_perf = PerformanceTimer().start()
            train_loss, train_accuracy = trainer.run()
            eval_loss, eval_accuracy = evaluator.run()
            roi = 1
            # roi = roi_evaluator.run()
            epoch_perf.stop()

            if eval_loss < self._best_loss:
                self._best_loss = eval_loss
                self.save_network()

            self._writer.add_scalar("Loss/train", train_loss, self._epoch)
            self._writer.add_scalar("Accuracy/train", train_accuracy, self._epoch)
            self._writer.add_scalar("Loss/eval", eval_loss, self._epoch)
            self._writer.add_scalar("Accuracy/eval", eval_accuracy, self._epoch)
            self._writer.add_scalar("Roi/eval", roi, self._epoch)
            print(f"[Epoch {self._epoch}] "
                  f"Train Loss: {train_loss:.4f}, Acc {train_accuracy:.2%}, "
                  f"Eval Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.2%}, "
                  f"Eval Roi: {roi:.2%}, "
                  f"Elapsed: {epoch_perf:.2f}s")


class Helper:
    @staticmethod
    def get_weights(dataset):
        # targets = np.array([self.get_label(x) for x in self.data_frames])
        occurrences = dataset.labels.cuda().bincount()
        # occurrences.resize(5)
        weights = len(dataset.labels) / (len(occurrences) * occurrences)
        return weights

    @staticmethod
    def get_weights(samples):
        occurrences = np.bincount(samples.labels)
        weights = len(samples.labels) / (len(occurrences) * occurrences)
        return weights

    @staticmethod
    def get_in_shape(sample: Sample):
        in_shape = np.asarray(sample.feature).shape
        print("estimated in shape: ", in_shape)
        return in_shape


class Runner:
    @abstractmethod
    def run(self):
        raise NotImplementedError


class ROIEvaluator(Runner):
    def __init__(self, frames: pd.DataFrame, converter: DataFrameSampleConverter, network: nn.Module,
                 device: torch.device):
        self.frames = frames
        self.convertor = converter
        self.network = network
        self.device = device

    def run(self):
        roi = 1
        self.network.eval()
        with torch.no_grad():
            hidden = None
            for frame in self.frames:
                if frame.close_price == 0:
                    continue
                next_frame = self.frames.get_next(frame)
                if next_frame.close_price == 0:
                    continue

                feature = self.convertor.get_frame_feature(frame)
                features = torch.as_tensor(np.array([[feature]]), dtype=torch.float, device=self.device)
                values, hidden = self.network(features, hidden)
                predicts = F.softmax(values, dim=1).argmax(dim=1)
                result = predicts[0].item()
                if result == 1:
                    if next_frame.high_price / frame.close_price >= 1.01:
                        roi *= 1.01
                    else:
                        roi *= next_frame.close_price / frame.close_price

        return roi


class Evaluator(Runner):
    def __init__(self, frames: pd.DataFrame, converter: DataFrameSampleConverter, network: nn.Module,
                 device: torch.device, steps: int = 1000):
        self.frames = frames
        self.convertor = converter
        self.network = network
        self.device = device
        self.steps = steps
        self.samples = Samples(converter.get_samples(frames))
        print(f"{type(self).__name__} samples:", len(self.samples))

        self.dataset = MarketDataset(self.samples)
        self.weights = Helper.get_weights(self.samples)
        self.batch_size = 128
        sampler = WeightedRandomSampler(self.weights[self.dataset.labels], self.steps * self.batch_size)
        # sampler = RandomSampler(train_dataset)
        self.loader = DataLoader(self.dataset,
                                 sampler=sampler,
                                 batch_size=self.batch_size,
                                 pin_memory=True,
                                 num_workers=0)

    def run(self):
        current_loss = 0
        total_data = 0
        correct_count = 0
        self.network.eval()
        iterator = iter(self.loader)
        with torch.no_grad():
            for step in range(self.steps):
                features, labels = next(iterator)
                features = features.to(self.device)
                labels = labels.to(self.device)
                values, _ = self.network(features)
                loss = nn.CrossEntropyLoss(weight=None, reduction="sum")(values, labels)
                predicts = F.softmax(values, dim=1).argmax(dim=1)
                correct_count += torch.eq(labels, predicts).count_nonzero()
                current_loss += loss
                total_data += len(features)

        average_loss = current_loss / total_data
        accuracy = correct_count / total_data
        return average_loss, accuracy


class Trainer(Runner):
    def __init__(self, frames: pd.DataFrames, converter: DataFrameSampleConverter, network: nn.Module, optimizer,
                 device: torch.device, steps: int = 1000):
        self.frames = frames
        self.convertor = converter
        self.network = network
        self.optimizer = optimizer
        self.device = device
        self.steps = steps
        self.samples = Samples(converter.get_samples(frames))
        print(f"{type(self).__name__} samples:", len(self.samples))
        self.dataset = MarketDataset(self.samples)
        self.weights = Helper.get_weights(self.samples)
        print(self.weights)
        self.batch_size = 128
        sampler = WeightedRandomSampler(self.weights[self.dataset.labels], self.batch_size * self.steps)
        # sampler = RandomSampler(train_dataset)
        self.loader = DataLoader(self.dataset,
                                 sampler=sampler,
                                 batch_size=self.batch_size,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=0)

    def run(self):
        current_loss = 0
        total_data = 0
        correct_count = 0
        self.network.train()
        iterator = iter(self.loader)
        for step in range(self.steps):
            features, labels = next(iterator)
            self.optimizer.zero_grad(set_to_none=True)

            features = features.to(self.device)
            labels = labels.to(self.device)
            values, _ = self.network(features)
            loss = nn.CrossEntropyLoss(weight=None, reduction="sum")(values, labels)
            loss.backward()
            # nn.utils.py.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
            self.optimizer.step()

            # Calculate Accuracy
            with torch.no_grad():
                predicts = F.softmax(values, dim=1).argmax(dim=1)
                correct_count += torch.eq(labels, predicts).count_nonzero()
                current_loss += loss
                total_data += len(features)

        average_loss = current_loss / total_data
        accuracy = correct_count / total_data
        return average_loss, accuracy


def main():
    register_signals()
    config = Config()
    crypto = Crypto(config)
    trainer = MarketAI(crypto, config)
    trainer.init()
    trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
