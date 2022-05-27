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
from torch.utils.data import Dataset, DataLoader
from tensorboard import program

warnings.filterwarnings("ignore", category=DeprecationWarning)

tokens = [
    "BUSD",
    "USDT",
    "USDC"
    "BTC",
    "ETH",
    "BNB",
]
token_dict = {}
for token_id, symbol in enumerate(tokens):
    token_dict[symbol] = token_id

markets = [
    ("BTC", "USDT"),
    ("BTC", "BUSD"),
    ("ETH", "USDT"),
    ("ETH", "BUSD"),
    ("BNB", "USDT"),
    ("BNB", "BUSD"),
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
        # self.features = torch.as_tensor(np.array([x["feature"] for x in data]), dtype=torch.float, device=device)
        # self.targets = torch.as_tensor(np.array([x["result"] for x in data]), dtype=torch.long, device=device)
        self.features = torch.as_tensor(np.array([x["feature"] for x in data]), dtype=torch.float)
        self.targets = torch.as_tensor(np.array([x["target"] for x in data]), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class CryptoDataLoader:
    def __init__(self, dataset, batch_size, pin_memory=False, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        if pin_memory and torch.cuda.is_available():
            self.pin_memory_device = torch.device("cuda")
        else:
            raise RuntimeError

    def __iter__(self):
        return CryptoDataLoaderIterator(self)


class CryptoDataLoaderIterator:
    def __init__(self, loader):
        self.loader = loader
        self.index = 0
        self.len = len(loader.dataset)
        self.batch_size = loader.batch_size
        self.indices = torch.randperm(self.len) if self.loader.shuffle else torch.arange(self.len)

    def __next__(self):
        if self.index >= self.len:
            raise StopIteration
        start_index = self.index
        self.index = min(self.index + self.batch_size, self.len)
        data = self.loader.dataset[self.indices[start_index: self.index]]
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
        hidden_notes = 128
        self.layers = nn.Sequential(
            nn.Linear(in_nodes, hidden_notes),
            nn.ReLU(),
            nn.Linear(hidden_notes, hidden_notes),
            nn.ReLU(),
            nn.Linear(hidden_notes, out_nodes)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-7)
        # self.optimizer = optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
        # self.schedular = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def forward(self, x):
        x = self.layers(x)
        # x = F.softmax(x, dim=-1)
        return x


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
    # samples
    print("[Creating Samples]")
    samples = []
    try:
        with open('data/samples.dat', 'rb') as sample_file:
            samples = pickle.load(sample_file)
            print("data file load successfully.")
    except:
        # loading data
        data = collections.defaultdict(dict)
        dt_format = datetime.strptime("01/01/2010", '%d/%m/%Y')
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
                        time = math.floor(dt_format.timestamp() * 1000)
                        # init market data
                        market_data = data[token_dict[token1]][token_dict[token2]] = {}
                        print(market_data)
                        while datetime.now().timestamp() > time / 1000:
                            lines = client.klines(market_id, "5m", startTime=time, limit=1000)
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

        data_len = sum([sum([len(market_data) for market_data in token1_market_data.values()]) for token1_market_data in data.values()])
        print("Data:", data_len)

        with open('data/samples.dat', 'wb') as sample_file:
            for token1, token1_market_data in data.items():
                for token2, market_data in token1_market_data.items():
                    for row in market_data.values():
                        try:
                            change_rate = (row["high_price"] - row["open_price"]) / row["open_price"]
                            samples.append({
                                "feature": get_feature(token1, token2, row, market_data),
                                "target": 1 if change_rate >= 0.01 else 0
                            })
                        except Exception as e:
                            # print(str(e))
                            pass
            pickle.dump(samples, sample_file)
    print("Samples:", len(samples))

    # samples = samples[:100000]
    random.Random(880712).shuffle(samples)
    split_index = math.floor(len(samples) * 0.9)
    training_samples = samples[:split_index]
    eval_samples = samples[split_index:]
    train_dataset = CryptoDataset(training_samples, device)
    train_dataloader = CryptoDataLoader(train_dataset, batch_size=256, pin_memory=True, shuffle=True, num_workers=0)
    test_dataset = CryptoDataset(eval_samples, device)
    eval_dataloader = CryptoDataLoader(test_dataset, batch_size=256, pin_memory=True, num_workers=0)

    print("Training samples:", len(training_samples))
    print("Eval samples:", len(eval_samples))

    # take one sample to create the network.
    network_path = "data/network.pt"
    train_features, train_labels = next(iter(train_dataloader))
    network = Network(len(train_features[0]), 2).to(device)
    epoch = 0

    try:
        checkpoint = torch.load(network_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        latest_loss = checkpoint["loss"]
        print(f"Network loaded with {epoch} trained, latest loss {latest_loss:.4f}")
    except:
        print("Failed to load the network.")

    # # push samples to tensors
    print("[Calculating Weights]")
    # occurrences = np.bincount([x["result"] for x in training_samples])
    occurrences = train_dataset.targets.bincount()
    # occurrences.resize(5)
    weights = train_dataset.targets.size(dim=0) / (occurrences.size(dim=0) * occurrences).to(device)
    print(weights)

    best = float('inf')
    while True:
        epoch += 1
        epoch_perf = PerformanceTimer().start()

        train_perf = PerformanceTimer().start()
        train_loss, train_accuracy = run("train", network, train_dataloader, device, weights)
        train_perf.stop()

        eval_perf = PerformanceTimer().start()
        eval_loss, eval_accuracy = run("eval", network, eval_dataloader, device, weights)
        eval_perf.stop()

        if eval_loss < best:
            best = eval_loss
            checkpoint = {
                "epoch": epoch,
                "loss": train_loss,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": network.optimizer.state_dict()
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


def run(mode, network, dataloader, device, weights):
    #
    # result = [{
    #     "total": 0,
    #     "correct": 0
    # } for i in range(0, 2)]
    if mode == "train":
        network.train()
    else:
        network.eval()
    current_loss = 0
    total_data = 0
    correct_count = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        probs = network(features)

        if mode == "eval":
            probs = probs.detach()

        loss = nn.CrossEntropyLoss(weight=weights, reduction="sum")(probs, labels)

        if mode == "train":
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
            network.optimizer.step()

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

    # if mode == "train":
    #     network.optimizer.step()
        # network.schedular.step()
    # for i, row in enumerate(result):
    #     print(mode, i, row["correct"], row["total"], row["correct_rate"])
    average_loss = current_loss / total_data
    accuracy = correct_count / total_data
    return average_loss, accuracy


def get_feature(token1, token2, row, data):
    price_multiplier = 10000
    # state_parts = [binary(math.floor(row["open_price"] * price_multiplier), 64)]
    open_time = datetime.fromtimestamp(row["open_time"])
    feature = [
        token1,
        token2,
        open_time.year,
        open_time.month,
        open_time.day,
        open_time.hour,
        open_time.minute
    ]
    for i in range(25):
        target_open_time = datetime.fromtimestamp(row["open_time"]) - timedelta(minutes=(1 + i) * 5)
        target_open_time_timestamp = target_open_time.timestamp()
        if target_open_time_timestamp not in data:
            raise LookupError(target_open_time)
        target_data = data[target_open_time_timestamp]

        # state_parts.append(binary(target_open_time.month, 4))
        # state_parts.append(binary(target_open_time.day, 8))
        # state_parts.append(binary(target_open_time.hour, 4))
        # state_parts.append(binary(target_open_time.minute, 8))
        # state_parts.append(binary(math.floor(target_data["open_price"] * price_multiplier), 64))
        # state_parts.append(binary(math.floor(target_data["high_price"] * price_multiplier), 64))
        # state_parts.append(binary(math.floor(target_data["low_price"] * price_multiplier), 64))
        # state_parts.append(binary(math.floor(target_data["volume"] * price_multiplier), 64))
        feature.append(target_open_time.year)
        feature.append(target_open_time.month)
        feature.append(target_open_time.day)
        feature.append(target_open_time.hour)
        feature.append(target_open_time.minute)
        feature.append(target_data["open_price"] / row["open_price"])
        feature.append(target_data["high_price"] / row["open_price"])
        feature.append(target_data["low_price"] / row["open_price"])
        feature.append(target_data["volume"] / row["volume"])
    # return np.concatenate(state_parts)
    return feature


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
