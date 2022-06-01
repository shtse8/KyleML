from binance.spot import Spot
from datetime import datetime, timedelta

import random
import logging
import math
import warnings
import collections
import sys
import signal
import asyncio
import numpy as np
import torch
import pickle
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
from tensorboard import program

warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 20200512

tokens = [
    "BUSD",
    "USDT",
    "USDC",
    "BTC",
    "ETH",
    "BNB",
]
token_dict = dict(zip(tokens, range(len(tokens))))
print(token_dict)
markets = [
    ("BTC", "USDT"),
    # ("BTC", "BUSD"),
    # ("ETH", "USDT"),
    # ("ETH", "BUSD"),
    # ("BNB", "USDT"),
    # ("BNB", "BUSD"),
]


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
        return self.elapsed()

    def __str__(self):
        return str(self.__repr__())

    def __format__(self, format_spec):
        return format(self.elapsed(), format_spec)


class CryptoDataset(Dataset):
    def __init__(self, data, device):
        self.features = torch.as_tensor(np.array([x[0] for x in data]), dtype=torch.float, device=device)
        self.targets = torch.as_tensor(np.array([x[1] for x in data]), dtype=torch.long, device=device)

    def to(self, device: torch.device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        return self

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class CryptoDataLoader:
    def __init__(self,
                 dataset,
                 batch_size,
                 pin_memory=False,
                 shuffle=False,
                 num_workers=0,
                 sampler=None,
                 generator=None,
                 drop_last=False):
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.sampler = sampler
        self.generator = generator
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        if pin_memory:
            if torch.cuda.is_available():
                self.pin_memory_device = torch.device("cuda")
            else:
                raise RuntimeError

    def __iter__(self):
        return CryptoDataLoaderIterator(self)


class CryptoDataLoaderIterator:
    def __init__(self, loader):
        # self.index = 0
        # self.len = len(loader.dataset)
        self.loader = loader
        self.batch_size = self.loader.batch_size
        if self.loader.sampler is not None:
            self.indices = self.loader.sampler
        elif self.loader.shuffle:
            self.indices = RandomSampler(self.loader.dataset, generator=self.loader.generator)
        else:
            self.indices = SequentialSampler(self.loader.dataset)
        # elif self.loader.shuffle:
        #     self.indices = torch.randperm(self.len)
        # else:
        #     self.indices = torch.arange(self.len)
        self.index_iter = iter(self.indices)
        self.drop_last = self.loader.drop_last

    def __next__(self):
        # if self.index >= self.len:
        #     raise StopIteration
        # if self.loader.drop_last and self.len - self.index < self.loader.batch_size:
        #     raise StopIteration
        # start_index = self.index
        # self.index = min(self.index + self.batch_size, self.len)
        # indices = self.indices[start_index: self.index]
        indices = [next(self.index_iter) for _ in range(0, self.batch_size)]
        data = self.loader.dataset[indices]
        if self.loader.drop_last and len(data) < self.loader.batch_size:
            raise StopIteration
        if self.loader.pin_memory:
            data = tuple([sample.pin_memory(self.loader.pin_memory_device) for sample in data])
        return data


def binary(num, bits):
    return ((num & (1 << np.arange(bits))[::-1]) > 0).astype(int)


def batches(input_list, batch_size):
    # try:
    idx = 0
    while idx < len(input_list):
        yield input_list[idx: min(idx + batch_size, len(input_list))]
        idx += batch_size
    # except:
    #     result = []
    #     iterator = iter(input_list)
    #     while (x := next(iterator, None)) is not None:
    #         result.append(x)
    #         if len(result) >= batch_size:
    #             yield result
    #             result = []
    #     yield result


# def binary(x, bits):
#     mask = 2**torch.arange(bits).to(x.device, x.dtype)
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def signal_handler(sig, frame):
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


class Network(nn.Module):

    def __init__(self, in_nodes: int, out_nodes: int):
        super(Network, self).__init__()
        hidden_nodes = 128
        self.layers = nn.Sequential(
            nn.Linear(in_nodes, hidden_nodes),
            nn.BatchNorm1d(hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, out_nodes)
        )

    def forward(self, x):
        x = self.layers(x)
        # x = F.softmax(x, dim=-1)
        return x


def get_data_loaders(samples, device, threshold=0.8, batch_size=64, num_workers=0, pin_memory=False):
    # samples = samples[:100000]
    random.Random(seed).shuffle(samples)
    split_index = math.floor(len(samples) * threshold)
    training_samples = samples[:split_index]
    eval_samples = samples[split_index:]
    if pin_memory:
        device = torch.device("cpu")

    train_dataset = CryptoDataset(training_samples, device)
    occurrences = train_dataset.targets.bincount()
    # occurrences.resize(5)
    labels_weights = train_dataset.targets.size(dim=0) / (occurrences.size(dim=0) * occurrences)
    weights = labels_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=pin_memory,
                                  shuffle=False,
                                  sampler=sampler,
                                  num_workers=num_workers)
    test_dataset = CryptoDataset(eval_samples, device)
    eval_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 pin_memory=pin_memory,
                                 num_workers=num_workers)
    print("Training samples:", len(training_samples))
    print("Eval samples:", len(eval_samples))

    return train_dataloader, eval_dataloader


def process_sample(data):
    for token1, token1_market_data in data.items():
        for token2, market_data in token1_market_data.items():
            for row in market_data.values():
                try:
                    yield get_sample(token1, token2, row, market_data)
                except Exception as e:
                    pass


def get_data():
    data = collections.defaultdict(dict)
    start_time_datetime = datetime.strptime("01/01/2010", '%d/%m/%Y')
    try:
        with open('data/data.dat', 'rb') as data_file:
            data = pickle.load(data_file)
            print("data file load successfully.")
    except Exception as e:
        with open('data/data.dat', 'wb') as data_file:
            client = Spot()
            for token1, token2 in markets:
                try:
                    market_id = token1 + token2
                    print("Fetching ", market_id, " market")
                    time = math.floor(start_time_datetime.timestamp() * 1000)
                    # init market data
                    market_data = data[token_dict[token1]][token_dict[token2]] = {}
                    while datetime.now().timestamp() > time / 1000:
                        lines = client.klines(market_id, "5m", startTime=time, limit=1000)
                        print(lines)
                        for line in lines:
                            open_time = math.floor(line[0] / 1000)
                            row = {
                                "open_time": open_time,
                                "open_price": float(line[1]),
                                "high_price": float(line[2]),
                                "low_price": float(line[3]),
                                "close_price": float(line[4]),
                                "volume": float(line[5])
                            }
                            market_data[open_time] = row
                        print(time, len(market_data))

                        # last close time + 1s
                        time = lines[-1][6] + 1
                except Exception as e:
                    print("Error - ", str(e))
                    # logging.exception("An exception was thrown!")
            pickle.dump(data, data_file)

    return data



def get_samples():
    # samples
    print("[Creating Samples]")
    samples = []
    with open('data/samples.dat', 'a+b') as sample_file:
        try:
            sample_file.seek(0)
            samples = pickle.load(sample_file)
            if len(samples) <= 0:
                raise Exception("No samples fetched.")
            print("data file load successfully.")
        except Exception as e:
            print("Failed to load samples: " + str(e))
            # loading data
            data = get_data()
            data_len = sum([sum([len(market_data) for market_data in x.values()]) for x in data.values()])
            print("Data:", data_len)
            samples = list(process_sample(data))
            sample_file.seek(0)
            pickle.dump(samples, sample_file)
            sample_file.truncate()

    print("Samples:", len(samples))
    if len(samples) <= 0:
        raise Exception("No samples loaded.")
    return samples


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda} (Devices: {torch.cuda.device_count()})")
    if torch.backends.cudnn.enabled:
        # torch.backends.cudnn.benchmark = True
        print(f"CUDNN {torch.backends.cudnn.version()}")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', "runs"])
    url = tb.launch()
    print("tensorboard url:", url)

    writer = SummaryWriter()

    # Get server timestamp
    # print(client.time())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, eval_dataloader = get_data_loaders(get_samples(), device, batch_size=2048, pin_memory=False)

    # take one sample to create the network.
    network_path = "data/network.pt"
    train_features, train_labels = next(iter(train_dataloader))
    in_features = len(train_features[0])

    torch.manual_seed(seed)
    network = Network(in_features, 2).to(device)
    epoch = 0

    optimizer = optim.AdamW(network.parameters(), lr=1e-4)
    # optimizer = optim.SGD(network.parameters(), lr=1e-4, momentum=0.9)
    # schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    try:
        checkpoint = torch.load(network_path, map_location=device)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        latest_loss = checkpoint["loss"]
        print(f"Network loaded with {epoch} trained, latest loss {latest_loss:.4f}")
    except:
        print("Failed to load the network.")

    # # push samples to tensors
    print("[Calculating Weights]")
    # weights = train_dataloader.dataset.get_weights().to(device)
    weights = None
    print(weights)

    # print("lr", optimizer.param_groups[0]['lr'])
    # for g in optimizer.param_groups:
    #     g['lr'] = 1e-8

    best = float('inf')
    while True:

        # print("lr", optimizer.param_groups[0]['lr'])

        epoch += 1
        epoch_perf = PerformanceTimer().start()

        train_perf = PerformanceTimer().start()
        train_loss, train_accuracy = run("train", network, train_dataloader, optimizer, schedular, device, weights)
        train_perf.stop()

        eval_perf = PerformanceTimer().start()
        eval_loss, eval_accuracy = run("eval", network, eval_dataloader, optimizer, schedular, device, weights)
        eval_perf.stop()

        if eval_loss < best:
            best = eval_loss
            checkpoint = {
                "epoch": epoch,
                "loss": train_loss,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, network_path)

        epoch_perf.stop()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/eval", eval_loss, epoch)
        writer.add_scalar("Accuracy/eval", eval_accuracy, epoch)
        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f}, Acc {train_accuracy:.2%}, "
              f"Eval Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.2%}, "
              f"Elapsed: {epoch_perf:.2f}s")


def run(mode, network, dataloader, optimizer, schedular, device, weights=None):
    is_train = mode == "train"
    #
    # result = [{
    #     "total": 0,
    #     "correct": 0
    # } for i in range(0, 2)]
    current_loss = 0
    total_data = 0
    correct_count = 0

    scaler = torch.cuda.amp.GradScaler()

    network.train(is_train)
    with torch.set_grad_enabled(is_train):
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=False):
                probs = network(features)
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
                scaler.step(optimizer)

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

    # for i, row in enumerate(result):
    #     print(mode, i, row["correct"], row["total"], row["correct_rate"])
    average_loss = current_loss / total_data
    accuracy = correct_count / total_data

    if mode == "train":
        schedular.step(average_loss)

    return average_loss, accuracy


def get_one_feature(data):
    data_open_time = datetime.fromtimestamp(data["open_time"])
    return [
        data_open_time.year,
        data_open_time.month,
        data_open_time.day,
        data_open_time.hour,
        data_open_time.minute,
        data["open_price"],
        data["high_price"],
        data["low_price"],
        data["volume"]
    ]


def get_sample(token1, token2, data, market_data):
    feature = [
        token1,
        token2
    ]
    feature += get_one_feature(data)
    for i in range(25):
        target_open_time = datetime.fromtimestamp(data["open_time"]) - timedelta(minutes=(1 + i) * 5)
        target_open_time_timestamp = target_open_time.timestamp()
        if target_open_time_timestamp not in market_data:
            raise LookupError(target_open_time)
        target_data = market_data[target_open_time_timestamp]
        feature += get_one_feature(target_data)

    change_rate = (data["high_price"] - data["open_price"]) / data["open_price"]
    target = 1 if change_rate >= 0.01 else 0

    return feature, target

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
