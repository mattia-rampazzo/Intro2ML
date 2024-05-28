import argparse
import os
import yaml
import wandb
import torch
import train
import timm
from dataset import get_data
from utils import get_loss_function, get_num_classes, get_optimizer, set_to_finetune_mode

from model import CustomClassifier

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
    #batch_size_test = config["data"]["batch_size_test"]  # same for train
    #num_workers = config["data"]["num_workers"]  # dynamically obtained
    num_epochs = config["training"]["num_epochs"]
    backbone_name = config["backbone_name"]  # (timm) backbone
    dataset_name = config["dataset_name"]   # dataset
 
 
    print("Starting")
    print(f"Working on {device}")

    # load model
    print(f"Loading {backbone_name} from timm...")
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes = get_num_classes(dataset_name))

    # Freeze the parameters of the base model
    for param in backbone.parameters():
      param.requires_grad = False
    # reset old classifier
    num_in_features = backbone.get_classifier().in_features
    backbone.reset_classifier()
    # combine backbone and new classifier
    model = CustomClassifier(backbone, num_in_features, get_num_classes(dataset_name))    
    model = model.to(device)

    print("Load data")
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=True)
    #transforms = timm.data.create_transform(**data_config, is_training=False)
    train_loader, val_loader, test_loader = get_data(dataset_name, batch_size_train, transforms, val_split=0.2)
    print("Done")

    # Define loss and optimizer
    loss_function = get_loss_function()
    optimizer = get_optimizer(model)

    # Define folder to save model weights
    save_folder = os.path.join("trained_models", backbone_name, dataset_name)
    run_name = args.run_name

    print("Training...")
    train.train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_function,
        num_epochs,
        device,
        save_folder,
        run_name
    )
    print("Done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)