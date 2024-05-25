import argparse
import yaml
import wandb
import torch
import torchvision
import torch.nn as nn
import train
import timm
from dataset import get_data
from utils import get_loss_function, get_num_classes, get_optimizer, set_to_finetune_mode

import timm
import model

def main(args):
    
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load config options
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # wandb login and initialization
    if config["logger"]["wandb"]:
        wandb.login()
        wandb.init(project="Competition", 
                   config=config, 
                   name=f"{args.run_name}")
        
    # get train options
    batch_size_train = config["data"]["batch_size_train"]
    batch_size_test = config["data"]["batch_size_test"]
    num_workers = config["data"]["num_workers"]
    num_epochs = config["training"]["num_epochs"]
    backbone_name = config["backbone"]  # (timm) backbone
    dataset_name = config["dataset_name"]   # dataset
 
 
    print("Starting")
    print(f"Working on {device}")

    # load model
    print(f"Loading {backbone_name} from timm...")
    encoder = timm.create_model(backbone_name, pretrained=True, num_classes = get_num_classes(dataset_name)) #feature_extractor=True)
    encoder.head = nn.Identity()
    encoder = encoder.to(device)
    # freeze all the layers up to last
    encoder = set_to_finetune_mode(encoder)

    # missing something here
    classifier = model.LabelPredictor().to(device)
    discriminator = model.DomainClassifier().to(device)
    print("Done")

    print("Load data")
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(encoder)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)

    train_loader, val_loader, test_loader = get_data(dataset_name, batch_size_train, transforms, num_workers, val_split=0.2)
    print("Done")

    # Define loss and optimizer
   # loss_function = get_loss_function()
   # optimizer = get_optimizer(encoder)

    # Define folder to save model weights
    save_folder = f"trained_models/{backbone_name}/{dataset_name}"
    run_name = args.run_name

    print("Training...")
    train.dann(
        encoder,
        classifier,
        discriminator,
        train_loader,
        train_dataset,
      #  val_loader,
        test_loader,
        test_dataset,
      #  optimizer,
      #  loss_function,
        num_epochs,
        device,
      #  save_folder,
      #  run_name
      config=config
    )
    print("Done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)