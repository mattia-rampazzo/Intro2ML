import os
import json
import torch
import yaml
import timm
import argparse
import requests
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from model import CustomClassifier
from utils import get_num_classes
from http.client import responses


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)
        self.image_paths = [os.path.join(root_dir, img_name) for img_name in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]  # Return image and its name


def main(args):
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config options
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Get test options
    batch_size_test = config["data"]["batch_size_test"]
    backbone_name = config["backbone_name"]
    dataset_name = config["dataset_name"]

    print("Starting")
    print(f"Working on {device}")

    # Load model
    print(f"Loading {backbone_name} from timm...")
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=get_num_classes(dataset_name))

    # Reset old classifier
    num_in_features = backbone.get_classifier().in_features
    backbone.reset_classifier(0)
    # Combine backbone and new classifier
    model = CustomClassifier(backbone, num_in_features, get_num_classes(dataset_name))

    # Load finetuned weights
    save_folder = os.path.join("trained_models", backbone_name, dataset_name)
    run_name = args.run_name
    file_path = os.path.join(save_folder, f"{run_name}.pt")
    state_dict = torch.load(file_path)
    model.classifier.load_state_dict(state_dict)

    # Set to eval mode
    model.eval()
    model = model.to(device)

    print("Load data")
    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    test_transforms = timm.data.create_transform(**data_config, is_training=False)
    test_data = TestDataset(os.path.join("data", "competition_data", "test"), test_transforms)

    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    print("Done")

    # Initialize an empty dictionary to store predictions
    predictions_dict = {}

    folder = ImageFolder(os.path.join("data", "competition_data", "train"))
    class_names = folder.classes

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, img_names) in enumerate(test_loader):
            # Load data into GPU
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Store predictions in the dictionary
            for img_name, prediction in zip(img_names, predicted):
                pred = prediction.item()  # Ensure prediction is a scalar
                predictions_dict[img_name] = class_names[pred].split("_")[0]

    # Output the predictions dictionary
    for img_name, prediction in predictions_dict.items():
        print(f'Image: {img_name}, Prediction: {prediction}')

    res = {
        "images": predictions_dict,
        "groupname": "BDV3000"
    }
    submit(res)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)