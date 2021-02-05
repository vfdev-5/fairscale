# adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from fairscale.optim.oss import OSS
from torch.nn.parallel import DistributedDataParallel as DDP

WORLD_SIZE = 2

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from helpers import dist_init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(rank, model, train_loader, num_epochs):
    ##############
    # SETUP
    dist_init(rank, WORLD_SIZE)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = OSS(
        params=model.parameters(),
        optim=torch.optim.Adadelta,
        lr=1e-4
    )

    training_start = time.monotonic()

    loss_fn = nn.CrossEntropyLoss()
    ##############

    model.train()
    measurements = []
    for epoch in range(num_epochs):
        epoch_start = time.monotonic()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)

            model.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{num_epochs} ")
        epoch_end = time.monotonic()

    torch.cuda.synchronize(rank)
    training_stop = time.monotonic()
    print("Total Time:", training_stop - training_start)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example with OSS")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=4, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True},)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    model = Net()

    mp.spawn(
        train, args=(model, train_loader, args.epochs), nprocs=WORLD_SIZE, join=True,
    )


if __name__ == "__main__":
    main()
