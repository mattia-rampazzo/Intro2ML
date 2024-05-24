import os
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.utils import download_file_from_google_drive

from utils import dataset_map

def get_data(dataset_name, batch_size, transforms, num_workers, val_split=0.2):
    download_path = os.path.join("data", dataset_name)

    # Load dataset
    if dataset_name in dataset_map:
        dataset_class = dataset_map[dataset_name]

        # if dataset_name=="tiny_imagenet":
        #     Dataset classes with split string parameter accepting train, val
        #     full_training_data = dataset_class(root=download_path, split="train", transform=transforms, download=True)
        #     test_data = dataset_class(root=download_path, split="val", transform=transforms, download=True)

        if dataset_name=="cub2011" or dataset_name=="dogs" or dataset_name=="food":
            if dataset_name=="food":
                # Dataset classes with split string parameter accepting train, val and test
                full_training_data = dataset_class(root=download_path, split="train", transform=transforms, download=True)
                test_data = dataset_class(root=download_path, split="test", transform=transforms, download=True)
            else:
                # Dataset classes with train boolean parameter
                full_training_data = dataset_class(root=download_path, train=True, transform=transforms, download=True)
                test_data = dataset_class(root=download_path, train=False, transform=transforms, download=True)
            # Create train and validation splits
            num_samples = len(full_training_data)
            training_samples = int(num_samples * (1 - val_split) + 1)
            validation_samples = num_samples - training_samples
            training_data, validation_data = random_split(full_training_data, [training_samples, validation_samples])
        else:
            # Dataset classes with split string parameter accepting train, val and test
            training_data = dataset_class(root=download_path, split="train", transform=transforms, download=True)
            validation_data = dataset_class(root=download_path, split="val", transform=transforms, download=True)
            test_data = dataset_class(root=download_path, split="test", transform=transforms, download=True)
    
    # special case for cars...
    else: 
        if dataset_name=="cars":
            # Datasets with ImageFolder
            train_path = os.path.join(download_path, 'car_data/car_data/train')
            test_path = os.path.join(download_path, 'car_data/car_data/test')

            archive_id = "1gwDRdAs9v39gyEeN3uWottjh4mXgsN-9" # GDrive id of the archive

            if not(os.path.exists(os.path.join(download_path, 'stanford_cars.zip'))):
                download_file_from_google_drive(archive_id, download_path, "stanford_cars.zip")
            if not(os.path.exists(train_path) and os.path.exists(test_path)):
                print("Extracting...")
                extract_archive(os.path.join(download_path, 'stanford_cars.zip'))
                
            full_training_data = torchvision.datasets.ImageFolder(root=train_path, transform=transforms)
            test_data = torchvision.datasets.ImageFolder(root=test_path, transform=transforms)
            # Create train and validation splits
            num_samples = len(full_training_data)
            training_samples = int(num_samples * (1 - val_split) + 1)
            validation_samples = num_samples - training_samples
            training_data, validation_data = random_split(full_training_data, [training_samples, validation_samples])
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # Print some stats
    print(f"# of training samples: {len(training_data)}")
    print(f"# of validation samples: {len(validation_data)}")
    print(f"# of test samples: {len(test_data)}")

    # Initialize dataloaders
    train_loader = DataLoader(training_data, batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader