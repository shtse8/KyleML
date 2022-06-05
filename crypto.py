from __future__ import annotations

import asyncio
import math
import pickle
import signal
import sys
import warnings
from abc import abstractmethod, ABC
from datetime import datetime, timedelta
from enum import Enum, auto
# from binance import AsyncClient
from os.path import exists
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from binance.spot import Spot
from tensorboard import program
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler

# from typing import Self

warnings.filterwarnings("ignore", category=DeprecationWarning)


class DataFrame:
    def __init__(self,
                 open_time: int,
                 open_price: float = 0,
                 high_price: float = 0,
                 low_price: float = 0,
                 close_price: float = 0,
                 volume: float = 0,
                 close_time: int = 0,
                 quote_asset_volume: float = 0,
                 number_of_trades: int = 0):
        self.open_time = open_time
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.close_time = close_time
        self.quote_asset_volume = quote_asset_volume
        self.number_of_trades = number_of_trades

    # def __repr__(self):
    #     return f"{type(self).__name__}({str(self)})"

    def __str__(self) -> str:
        return f"{self.open_time}, {self.open_price:.4}, {self.high_price:.4}, {self.low_price:.4}, {self.close_price:.4}, {self.volume:.4}, {self.close_time}, {self.quote_asset_volume}, {self.number_of_trades}"


class DataFrames:
    def __init__(self, interval: int = 300):
        self.frame_dict = {}
        self.interval = interval
        self.start_time: int = 0
        self.end_time: int = 0

    def add(self, frame: DataFrame):
        if self.start_time != 0 and (frame.open_time - self.start_time) % self.interval != 0:
            warnings.warn(f"the open time is not fit for this frame set. "
                          f"{frame.open_time} - {self.start_time} = "
                          f"{frame.open_time - self.start_time}", UserWarning)
        #     raise ValueError(f"the open time is not fit for this frame set. "
        #                      f"{frame.open_time} - {self.start_time} = "
        #                      f"{frame.open_time - self.start_time}")

        self.frame_dict[frame.open_time] = frame

        # update start time and end time
        if self.start_time == 0:
            self.start_time = frame.open_time
            self.end_time = frame.open_time
        else:
            if frame.open_time < self.start_time:
                self.start_time = frame.open_time
            if frame.open_time > self.end_time:
                self.end_time = frame.open_time

    def ensure_valid_timestamp(self, timestamp: int):
        if (timestamp - self.start_time) % self.interval != 0:
            raise Exception("Timestamp is not match for this data frames.")

    def __len__(self) -> int:
        if self.end_time < self.start_time:
            return 0
        return (self.end_time - self.start_time) // self.interval + 1

    def splice(self, start_index: int, end_index: int = 0) -> DataFrames:
        new_data_frames = DataFrames()
        end_index = max(len(self), end_index)
        for i in range(start_index, end_index):
            new_data_frames.add(self[i])
        return new_data_frames

    def __getitem__(self, key) -> DataFrame | DataFrames:
        # return multiple frames
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            new_data_frames = DataFrames()
            for i in range(*key.indices(len(self))):
                new_data_frames.add(self[i])
            return new_data_frames
        # return multiple frames
        elif isinstance(key, list):
            new_data_frames = DataFrames()
            for i in key:
                new_data_frames.add(self[i])
            return new_data_frames
        # return single frame
        elif isinstance(key, int):
            # elif isinstance(key, int | float):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)

            open_time = self.start_time + key * self.interval
            return self.from_timestamp(open_time)
        # return single frame
        elif isinstance(key, datetime):
            return self.from_timestamp(int(key.timestamp()))  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self) -> DataFrameIterator:
        return DataFrameIterator(self)

    def from_timestamp(self, timestamp: int) -> DataFrame:
        self.ensure_valid_timestamp(timestamp)
        return self.frame_dict[timestamp] if timestamp in self.frame_dict else DataFrame(timestamp)


class DataFrameIterator(Iterator):
    def __init__(self, data_frames: DataFrames):
        self.data_frames = data_frames
        self.index = 0

    def __next__(self):
        if self.index >= len(self.data_frames):
            raise StopIteration
        frame = self.data_frames[self.index]
        self.index += 1
        return frame


class Token:
    id: int
    symbol: str

    def __init__(self, id: int, symbol: str):
        self.id = id
        self.symbol = symbol

    def __eq__(self, other):
        return self.symbol == other.symbol


class MarketAgent(ABC):
    name: str = ""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_frames(self, token1: Token, token2: Token) -> DataFrames:
        raise NotImplementedError


class BinanceMarketAgent(MarketAgent):
    _client: Spot = Spot()

    def __init__(self):
        super().__init__("Binance")

    def get_frames(self, token1: Token, token2: Token) -> DataFrames:
        data_frames = DataFrames()
        start_time_datetime = datetime.strptime("01/01/2010", '%d/%m/%Y')
        market_id = token1.symbol + token2.symbol
        print("Fetching ", market_id, " market")
        time = math.floor(start_time_datetime.timestamp() * 1000)
        while datetime.now().timestamp() > time / 1000:
            lines = self._client.klines(market_id, "5m", startTime=time, limit=1000)
            for line in lines:
                open_time = line[0] // 1000
                open_price = float(line[1])
                high_price = float(line[2])
                low_price = float(line[3])
                close_price = float(line[4])
                volume = float(line[5])
                close_time = line[6] // 1000
                quote_asset_volume = float(line[7])
                number_of_trades = int(line[8])
                frame = DataFrame(open_time,
                                  open_price,
                                  high_price,
                                  low_price,
                                  close_price,
                                  volume,
                                  close_time,
                                  quote_asset_volume,
                                  number_of_trades)
                data_frames.add(frame)
            print(time, len(data_frames))

            # last close time + 1s
            time = lines[-1][6] + 1

        print("Returning")
        return data_frames


class Market:
    _data_frames: DataFrames = None

    def __init__(self, agent: MarketAgent, token1: Token, token2: Token):
        self.token1 = token1
        self.token2 = token2
        self.agent = agent
        self.cache_path = f'data/market.{self.token1.symbol}{self.token2.symbol}.pickle'

    def _load_data(self):
        if not exists(self.cache_path):
            raise FileNotFoundError

        with open(self.cache_path, 'rb') as file:
            data = pickle.load(file)
            if data is None:
                raise ValueError
            return data

    def get_data(self, update: bool = False) -> DataFrames:
        data = None
        try:
            if not update:
                data = self._load_data()
        except (FileNotFoundError, ValueError, EOFError) as _:
            update = True

        if update:
            with open(self.cache_path, 'wb') as file:
                data = self.agent.get_frames(self.token1, self.token2)
                pickle.dump(data, file)

        return data

    @property
    def data_frames(self):
        if self._data_frames is None:
            self._data_frames = self.get_data()
        return self._data_frames


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
    seed: int = 880605
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


class PerformanceTimer:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = perf_counter()
        return self

    def stop(self):
        self.stop_time = perf_counter()
        return self

    def elapsed(self):
        return self.stop_time - self.start_time

    def __repr__(self):
        return f"{type(self).__name__}({str(self)})"

    def __str__(self):
        return f"{self.elapsed():.4}s"

    def __format__(self, format_spec):
        return format(self.elapsed(), format_spec)


class Samples:
    def __init__(self, features: Sequence = None, labels: Sequence = None):
        self._features = features if features is not None else []
        self._labels = labels if labels is not None else []
        if len(self._features) != len(self._labels):
            raise Exception(f"features and labels must have same number of elements. "
                            f"num_features={len(self.features)}, "
                            f"num_labels={len(self._labels)}")

    @property
    def features(self):
        return self._features

    @property
    def labels(self) -> Sequence[int]:
        return self._labels

    def __len__(self):
        return len(self._features)

    def __getitem__(self, item) -> (Sequence[float], Sequence[int]) | Samples:
        if isinstance(item, int):
            return self._features[item], self._labels[item]
        else:
            return Samples(self._features[item], self._labels[item])

    def __iter__(self):
        yield from iter([self[x] for x in range(len(self))])


class DataFramesSamples(Samples):
    def __init__(self, data_frames: DataFrames):
        self._data_frames = data_frames
        self._frame_features = {}
        super().__init__(
            features=[self.get_feature(x) for x in data_frames],
            labels=[self.get_label(x) for x in data_frames])

    def get_frame_feature(self, frame: DataFrame):
        data_open_time = datetime.fromtimestamp(frame.open_time)
        data_close_time = datetime.fromtimestamp(frame.close_time)
        return [
            data_open_time.year,
            data_open_time.month,
            data_open_time.day,
            data_open_time.hour,
            data_open_time.minute,
            frame.open_price,
            frame.high_price,
            frame.low_price,
            frame.close_price,
            frame.volume,
            data_close_time.year,
            data_close_time.month,
            data_close_time.day,
            data_close_time.hour,
            data_close_time.minute,
            frame.quote_asset_volume,
            frame.number_of_trades
        ]

    def get_cached_frame_feature(self, frame: DataFrame):
        if frame.open_time in self._frame_features:
            return self._frame_features[frame.open_time]
        self._frame_features[frame.open_time] = self.get_frame_feature(frame)
        return self._frame_features[frame.open_time]

    def get_feature(self, frame: DataFrame):
        # Converts the current data frame to a running frame at start
        running_frame_at_start = DataFrame(frame.open_time, frame.open_price)
        features = [self.get_cached_frame_feature(running_frame_at_start)]
        frame_open_datetime = datetime.fromtimestamp(frame.open_time)
        for i in range(100):
            running_frame_open_datetime = frame_open_datetime - timedelta(minutes=(1 + i) * 5)
            # running_frame_open_time = running_frame_open_datetime.timestamp()
            features.append(self.get_cached_frame_feature(self._data_frames[running_frame_open_datetime]))
        return np.asarray(features).flatten()

    def get_label(self, frame: DataFrame):
        if frame.open_price == 0:
            return 0
        change_rate = (frame.high_price - frame.open_price) / frame.open_price
        return 1 if change_rate >= 0.01 else 0


class SampleAnalyzer:
    def __init__(self, samples: Samples):
        self.samples = samples

    def get_in_nodes(self):
        # train_feature = self.get_feature(self.data_frames[0])
        in_nodes = len(self.samples[0][0])
        print("estimated in nodes: ", in_nodes)
        return in_nodes

    def get_out_nodes(self):
        out_nodes = max(self.samples.labels) + 1
        print("estimated out nodes: ", out_nodes)
        return out_nodes

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


# def binary(num, bits):
#     return ((num & (1 << np.arange(bits))[::-1]) > 0).astype(int)
#
#
# def batches(input_list, batch_size):
#     # try:
#     idx = 0
#     while idx < len(input_list):
#         yield input_list[idx: min(idx + batch_size, len(input_list))]
#         idx += batch_size
#     # except:
#     #     result = []
#     #     iterator = iter(input_list)
#     #     while (x := next(iterator, None)) is not None:
#     #         result.append(x)
#     #         if len(result) >= batch_size:
#     #             yield result
#     #             result = []
#     #     yield result


# def binary(x, bits):
#     mask = 2**torch.arange(bits).to(x.device, x.dtype)
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


def signal_handler(sig, frame):
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


class Network(nn.Module):

    def __init__(self, in_nodes: int, out_nodes: int):
        super(Network, self).__init__()
        hidden_nodes = 512
        self.layers = nn.Sequential(
            nn.Linear(in_nodes, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            # nn.Linear(256, 128),
            # nn.ELU(),
            # SwitchNorm1d(128),
            # nn.Linear(128, 64),
            # nn.ELU(),
            # SwitchNorm1d(64),
            nn.Linear(512, out_nodes)
        )

    def forward(self, x):
        x = self.layers(x)
        # x = F.softmax(x, dim=-1)
        return x


# def process_sample(data):
#     for token1, token1_market_data in data.items():
#         for token2, market_data in token1_market_data.items():
#             for row in market_data.values():
#                 try:
#                     yield get_sample(token1, token2, row, market_data)
#                 except Exception as e:
#                     pass


#
# def get_samples():
#     # samples
#     print("[Creating Samples]")
#     samples = []
#     with open('data/samples.dat', 'a+b') as sample_file:
#         try:
#             sample_file.seek(0)
#             samples = pickle.load(sample_file)
#             if samples is None:
#                 raise Exception("No samples fetched.")
#             print("data file load successfully.")
#         except Exception as e:
#             print("Failed to load samples: " + str(e))
#             # loading data
#             data = get_data()
#             data_len = sum([sum([len(market_data) for market_data in x.values()]) for x in data.values()])
#             print("Data:", data_len)
#             samples = list(process_sample(data))
#             sample_file.seek(0)
#             pickle.dump(samples, sample_file)
#             sample_file.truncate()
#
#     print("Samples:", len(samples))
#     if len(samples) <= 0:
#         raise Exception("No samples loaded.")
#     return samples

class RunMode(Enum):
    Train = auto()
    Eval = auto()


class Trainer:
    _writer: SummaryWriter = None
    _network: nn.Module = None
    _optimizer: Optimizer = None
    _schedular: any = None
    _epoch = 0
    _latest_loss = 0
    _best_loss = float("inf")
    _train_dataloader = None
    _eval_dataloader = None

    def __init__(self, crypto: Crypto, config: Config):
        self._crypto = crypto
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        self.check()
        self.launch_tensor_board()

    def create_network(self, in_nodes, out_nodes):
        torch.manual_seed(self._config.seed)
        self._network = Network(in_nodes, out_nodes).to(self._device)
        self._optimizer = optim.AdamW(self._network.parameters(), lr=1e-4)
        # optimizer = optim.SGD(network.parameters(), lr=1e-4, momentum=0.9)
        # schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self._schedular = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, verbose=True)
        self._epoch = 0

    def load_network(self):
        checkpoint = torch.load(self._config.network_path, map_location=self._device)
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        latest_loss = checkpoint["loss"]
        self._best_loss = latest_loss
        print(f"Network loaded with {epoch} trained, latest loss {latest_loss:.4f}")

    def save_network(self, loss):
        checkpoint = {
            "epoch": self._epoch,
            "loss": self._latest_loss,
            "model_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict()
        }
        torch.save(checkpoint, self._config.network_path)

    def split_samples(self,
                      data_frames: DataFrames,
                      threshold: float = 0.8) -> (Samples, Samples):
        samples = DataFramesSamples(data_frames)
        split_index = math.floor(len(samples) * threshold)
        training_samples = samples[:split_index]
        eval_samples = samples[split_index:]
        return training_samples, eval_samples

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

    def run(self):

        market = self._crypto.markets[0]

        # prepare samples
        train_samples, eval_samples = self.split_samples(market.data_frames)
        print("Training samples:", len(train_samples))
        print("Eval samples:", len(eval_samples))

        analyzer = SampleAnalyzer(train_samples)

        # init network
        try:
            self.load_network()
        except Exception as e:
            print("Failed to load the network.")
            self.create_network(analyzer.get_in_nodes(), analyzer.get_out_nodes())

        # Create pytorch dataset
        train_dataset = MarketDataset(train_samples)
        eval_dataset = MarketDataset(eval_samples)

        print("[Calculating Weights]")
        # weights = train_dataloader.dataset.get_weights().to(device)
        weights = torch.as_tensor(analyzer.get_weights(), device=self._device, dtype=torch.float)
        print(weights)

        sampler = WeightedRandomSampler(weights[train_dataset.labels], len(train_dataset.labels))
        # sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=sampler,
                                      batch_size=128,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=0)

        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=sampler,
                                     batch_size=128,
                                     pin_memory=True,
                                     num_workers=0)

        while True:
            # print("lr", optimizer.param_groups[0]['lr'])
            self._epoch += 1
            epoch_perf = PerformanceTimer().start()
            train_perf = PerformanceTimer().start()
            train_loss, train_accuracy = self.run_epoch(RunMode.Train, train_dataloader)
            train_perf.stop()

            eval_perf = PerformanceTimer().start()
            eval_loss, eval_accuracy = self.run_epoch(RunMode.Eval, eval_dataloader)
            eval_perf.stop()
            epoch_perf.stop()

            if eval_loss < self._best_loss:
                self._best_loss = eval_loss
                self.save_network(eval_loss)

            self._writer.add_scalar("Loss/train", train_loss, self._epoch)
            self._writer.add_scalar("Accuracy/train", train_accuracy, self._epoch)
            self._writer.add_scalar("Loss/eval", eval_loss, self._epoch)
            self._writer.add_scalar("Accuracy/eval", eval_accuracy, self._epoch)
            print(f"[Epoch {self._epoch}] "
                  f"Train Loss: {train_loss:.4f}, Acc {train_accuracy:.2%}, "
                  f"Eval Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.2%}, "
                  f"Elapsed: {epoch_perf:.2f}s")

    def run_epoch(self, mode: RunMode, dataloader: DataLoader, weights=None):
        is_train = mode == RunMode.Train
        # dataloader = self._train_dataloader if is_train else self._eval_dataloader

        #
        # result = [{
        #     "total": 0,
        #     "correct": 0
        # } for i in range(0, 2)]
        current_loss = 0
        total_data = 0
        correct_count = 0

        scaler = torch.cuda.amp.GradScaler()
        steps = 0
        self._network.train(is_train)
        with torch.set_grad_enabled(is_train):
            for features, labels in dataloader:
                if is_train:
                    self._optimizer.zero_grad(set_to_none=True)

                features = features.to(self._device)
                labels = labels.to(self._device)
                with torch.cuda.amp.autocast(enabled=False):
                    probs = self._network(features)
                    loss = nn.CrossEntropyLoss(weight=weights, reduction="sum")(probs, labels)

                if is_train:
                    scaler.scale(loss).backward()

                    # Unscales the gradients of optimizer's assigned params in-place
                    # scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    # nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                    # network.optimizer.step()
                    scaler.step(self._optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                # Calculate Accuracy
                with torch.no_grad():
                    predicts = F.softmax(probs, dim=1).argmax(dim=1)
                    # check equal between targets and predicts, then zip, then count unique
                    # we would like to do it in tensor to speed up
                    # rows, counts = torch.stack((labels, torch.eq(labels, predicts)), dim=1).unique(dim=0, return_counts=True)
                    # for (label, matched), count in zip(rows, counts):
                    #     stats = result[label]
                    #     stats["total"] += count.item()
                    #     if matched == 1:
                    #         stats["correct"] += count.item()
                    #     stats["correct_rate"] = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else float("nan")

                    correct_count += torch.eq(labels, predicts).count_nonzero()

                    current_loss += loss
                    total_data += len(features)

                steps += 1
                if steps >= 1000:
                    break

        # for i, row in enumerate(result):
        #     print(mode, i, row["correct"], row["total"], row["correct_rate"])
        average_loss = current_loss / total_data
        accuracy = correct_count / total_data

        # if mode == "train":
        #     schedular.step(average_loss)

        return average_loss, accuracy


def main():
    signal.signal(signal.SIGINT, signal_handler)

    config = Config()

    crypto = Crypto(config)
    trainer = Trainer(crypto, config)

    trainer.init()

    trainer.run()

    # Get server timestamp
    # print(client.time())


#
# def get_one_feature(data):
#     data_open_time = datetime.fromtimestamp(data["open_time"])
#     return [
#         data_open_time.year,
#         data_open_time.month,
#         data_open_time.day,
#         data_open_time.hour,
#         data_open_time.minute,
#         data["open_price"],
#         data["high_price"],
#         data["low_price"],
#         data["volume"]
#     ]
#
#
# def get_sample(token1, token2, data, market_data):
#     feature = [
#         token1,
#         token2
#     ]
#     feature += get_one_feature(data)
#     for i in range(25):
#         target_open_time = datetime.fromtimestamp(data["open_time"]) - timedelta(minutes=(1 + i) * 5)
#         target_open_time_timestamp = target_open_time.timestamp()
#         if target_open_time_timestamp not in market_data:
#             raise LookupError(target_open_time)
#         target_data = market_data[target_open_time_timestamp]
#         feature += get_one_feature(target_data)
#
#     change_rate = (data["high_price"] - data["open_price"]) / data["open_price"]
#     target = 1 if change_rate >= 0.01 else 0
#
#     return feature, target

# def get_data(date):

# class DataManager:
#     def __init__(self):
#         data = {}
#
#     def _

# api key/secret are required for user data endpoints
# client = Spot(key=binance_key, secret=binance_secret)

# Get account and balance information
# print(client.account())

# Post a new order
# params = {
#     'symbol': 'BTCUSDT',
#     'side': 'SELL',
#     'type': 'LIMIT',
#     'timeInForce': 'GTC',
#     'quantity': 0.002,
#     'price': 9500
# }
#
# response = client.new_order(**params)
# print(response)


if __name__ == "__main__":
    asyncio.run(main())
