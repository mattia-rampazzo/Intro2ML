import argparse
import os
import yaml
import wandb
import torch
import train
import timm
from dataset import get_data
from utils import get_loss_function, get_num_classes, get_optimizer, get_transform, get_scheduler

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
    backbone.reset_classifier(0)
    # combine backbone and new classifier
    model = CustomClassifier(backbone, num_in_features, get_num_classes(dataset_name))    
    model = model.to(device)

    print("Load data")
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = get_transform(data_config, True)
    #transforms = timm.data.create_transform(**data_config, is_training=False)
    train_loader, val_loader, test_loader = get_data(dataset_name, batch_size_train, transforms, val_split=0.2)
    print("Done")

    # Define loss and optimizer
    lr_begin = (batch_size_train / 256) * 0.1  # learning rate at begining
    momentum = 0.9
    wd = 5e-4
    loss_function = get_loss_function(get_num_classes(dataset_name))
    optimizer = get_optimizer(model, lr_begin , wd, momentum)
    scheduler = get_scheduler(optimizer)

    # Define folder to save model weights
    save_folder = os.path.join("trained_models", backbone_name, dataset_name)
    run_name = args.run_name

    use_amp = 2
    ##### Apex
    if use_amp == 1:  # use nvidia apex.amp
        print('\n===== Using NVIDIA AMP =====')
        from apex import amp

        net.cuda()
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    elif use_amp == 2:  # use torch.cuda.amp
        print('\n===== Using Torch AMP =====')
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()

    print("Training...")
    train.train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
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