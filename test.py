import torch
import yaml
import timm
import argparse
import os

from model import CustomClassifier
from utils import get_num_classes
from dataset import get_data
from utils import get_loss_function
from train import test_step

def main(args):
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load config options
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # get test options
    batch_size_test = config["data"]["batch_size_test"]  # same for train
    #num_workers = config["data"]["num_workers"]  # dynamically obtained
    backbone_name = config["backbone_name"]  # (timm) backbone
    dataset_name = config["dataset_name"]   # dataset
    print("Starting")
    print(f"Working on {device}")


    # load model
    print(f"Loading {backbone_name} from timm...")
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes = get_num_classes(dataset_name))

    # reset old classifier
    num_in_features = backbone.get_classifier().in_features
    backbone.reset_classifier(0)
    # combine backbone and new classifier
    model = CustomClassifier(backbone, num_in_features, get_num_classes(dataset_name)) 

    # load finetuned weights
    save_folder = os.path.join("trained_models", backbone_name, dataset_name)
    run_name = args.run_name
    file_path = os.path.join(save_folder, f"{run_name}.pt")
    state_dict = torch.load(file_path)
    model.classifier.load_state_dict(state_dict)

    # set to eval mode
    model.eval()
    model = model.to(device)

    print("Load data")
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    #transforms = timm.data.create_transform(**data_config, is_training=True)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    _, _, test_loader = get_data(dataset_name, batch_size_test, transforms, val_split=0.2)
    print("Done")


    print("Load data")
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    #transforms = timm.data.create_transform(**data_config, is_training=True)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    _, _, test_loader = get_data(dataset_name, batch_size_test, transforms, val_split=0.2)
    print("Done")


    print("Evaluating on test data...")
    # Define loss and optimizer
    loss_function = get_loss_function(get_num_classes(dataset_name))
    loss, acc, pred = test_step(model, test_loader, loss_function, device)

    print(f"Top 1 accuracy: {acc}")

    print("Done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)
