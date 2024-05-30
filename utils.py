import torch
import torchvision
import os
import timm

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

def set_to_finetune_mode(model, do_summary=False):

    classifier = model.default_cfg['classifier']
    # print(classifier)

    # Freeze all parameters except those in the classifier
    for name, param in model.named_parameters():
        if name.startswith(classifier):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if do_summary:
        from torchinfo import summary
        summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        ) 

    return model

def get_transforms(data_config, is_training):
    transforms = timm.data.create_transform(is_training=is_training, **data_config, auto_augment='rand-m9-n3-mstd0.5')
    return transforms


def get_optimizer(model):
    lr=0.001
    wd=0.0001
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def get_scheduler(optimizer):
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
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