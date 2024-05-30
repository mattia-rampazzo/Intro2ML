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


def get_optimizer(model, lr, wd, momentum=0.9):
    print(f"old lr:{lr}")
    lr=0.001
    wd=0.0001
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def get_loss_function(num_classes):
    ##### optimizer setting
    loss_function = LabelSmoothingLoss(
        classes=num_classes, smoothing=0.1
    )  # label smoothing to improve performance
    # loss_function = nn.CrossEntropyLoss()
    return loss_function

def get_scheduler(optimizer):
    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) # num_epochs
    return scheduler

def get_transform(data_config, is_training):
    print("Using standard trasforms")
    return timm.data.create_transform(**data_config, is_training=False, auto_augment='rand-m9-n3-mstd0.5')

    re_size = 512
    crop_size = 224

    if is_training==True:
        train_transform = transforms.Compose(
            [
                transforms.Resize((re_size, re_size)),
                transforms.RandomCrop(crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return train_transform
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((re_size, re_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return test_transform
    
def save_model(weights, save_folder, run_name):
    print("Saving model")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(
        weights,
        os.path.join(save_folder, f"{run_name}.pt")
    )