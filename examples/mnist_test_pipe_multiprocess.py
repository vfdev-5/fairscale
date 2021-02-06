# adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.pipe import MultiProcessPipe

WORLD_SIZE = 2

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from helpers import dist_init

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


def train(rank, model, train_loader, num_epochs):
    ##############
    # SETUP
    dist_init(rank, WORLD_SIZE)

    device = torch.device("cpu") if DEVICE == "cpu" else rank  # type:ignore

    # model = Pipe(
    #     model,
    #     balance=[6, 6],
    #     worker_map={0: "worker0", 1: "worker1"},  # Needed to convert ranks to RPC worker names
    #     input_device=device,
    # ).to(device)

    model = MultiProcessPipe(
        model,
        balance=[6, 6],
        worker_map={0: "worker0", 1: "worker1"},  # Needed to convert ranks to RPC worker names
        input_device=device,
    ).to(device)

    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=1e-4)

    training_start = time.monotonic()

    loss_fn = nn.CrossEntropyLoss()
    ##############

    model.train()
    measurements = []
    for epoch in range(num_epochs):
        epoch_start = time.monotonic()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # outputs and target need to be on the same device
            # forward step
            outputs = model(data.to(device))
            # compute loss
            if rank == 1:
                loss = loss_fn(outputs.to(device), target.to(device))
                # backward + optimize
                loss.backward()
                optimizer.step()
            else:
                model.back_helper(outputs)

        print(f"Epoch {epoch}/{num_epochs} ")
        epoch_end = time.monotonic()

    dist.rpc.shutdown()
    training_stop = time.monotonic()
    print("Total Time:", training_stop - training_start)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example with OSS")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update(
            {"num_workers": 1, "pin_memory": True, "shuffle": True},
        )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    model = net

    mp.spawn(
        train,
        args=(model, train_loader, args.epochs),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
