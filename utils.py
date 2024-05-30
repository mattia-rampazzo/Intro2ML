import torch
import torchvision
import os
import timm
import torch.nn as nn
from dogs import Dogs
from cub2011 import Cub2011
from LabelSmoothing import LabelSmoothingLoss

dataset_num_classes = {
    'aircraft': 100,
    'cars': 196,
    'cub2011': 200,
    'dogs': 120,
    'food': 101,
    'flowers': 102
    #'inat2017': None, # too big
    #'nabirds': 400, # no longer publicly accessible
    #'tiny_imagenet': 200 # too big + not a good benchmark
}

# Mapping dataset names to their corresponding classes
dataset_map = {
    'aircraft': torchvision.datasets.FGVCAircraft,
    'cub2011': Cub2011,
    'dogs': Dogs,
    'food': torchvision.datasets.Food101,
    'flowers': torchvision.datasets.Flowers102
    #'inat2017': INat2017,
    #'nabirds': NABirds,
    #'tiny_imagenet': TinyImageNet 
}

def get_num_classes(dataset_name):
    num_classes = 1000

    # Load dataset
    if dataset_name in dataset_num_classes:
        num_classes = dataset_num_classes[dataset_name]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return num_classes


def get_transforms(data_config, is_training):
    transforms = None
    if is_training:
        transforms = timm.data.create_transform(**data_config, is_training=is_training, auto_augment='rand-m9-mstd0.5')
    else:
        transforms = timm.data.create_transform(**data_config, is_training=is_training)
    return transforms


def get_optimizer(model):
    lr=0.01
    wd=0.0001
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def get_scheduler(optimizer):
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) # num_epochs
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    return scheduler

def get_loss_function(num_classes):
    ##### optimizer setting
    loss_function = LabelSmoothingLoss(
        classes=num_classes, smoothing=0.1
    )  # label smoothing to improve performance
    # loss_function = nn.CrossEntropyLoss()
    return loss_function

def save_model(weights, save_folder, run_name):
    print("Saving model")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(
        weights,
        os.path.join(save_folder, f"{run_name}.pt")
    )