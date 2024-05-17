import argparse
import yaml
import wandb
import torch
import torch.nn as nn

import train
from dataset import get_cub


import timm


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

def get_optimizer(model):
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer

def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function


def main(args):
    
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if config["logger"]["wandb"]:
        wandb.login()
        wandb.init(project="UDA", 
                   config=config, 
                   name=f"{args.run_name}")
        
       
    batch_size_train = config["data"]["batch_size_train"]
    # batch_size_test = config["data"]["batch_size_test"]
    # num_workers = config["data"]["num_workers"]
    num_epochs = config["training"]["num_epochs"]
    
    backbone_name = config["backbone"]  # (timm) backbone
    num_classes = config["num_classes"]   # out classes
 
    print("Let's start!")
    print(f"Working on {device}")

    # loading the model
    print(f"Loading {backbone_name} from timm")
    model = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    model = set_to_finetune_mode(model) # freeze all the layers up to last

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print("Done")
    

    print("Load data")
    train_loader, val_loader, test_loader = get_cub(batch_size_train, transforms, val_split=0.2)
    print("Done")


    # Define loss and optimizer
    loss_function = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training")
    train.train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_function,
        num_epochs,
        device
    )
    print("Done")





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)