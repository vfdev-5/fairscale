# adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from fairscale.nn import Pipe

net = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout2d(0.25),
    nn.Flatten(1),
    nn.Linear(9216, 128),
    nn.ReLU(),
    nn.Dropout2d(0.5),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1),
)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.to(device), target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=4, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {"batch_size": args.batch_size}
    kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True},)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    model = net
    pipe_model = Pipe(model, balance=[6, 6], devices=[0, 1], chunks=2)
    device = pipe_model.devices[0]

    optimizer = optim.Adadelta(pipe_model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        tic = time.perf_counter()
        train(pipe_model, device, train_loader, optimizer, epoch)
        toc = time.perf_counter()
        print(f">>> TRANING Time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()
