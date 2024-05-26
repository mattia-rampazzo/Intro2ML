import copy
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import test
import utils 
import numpy as np
from utils import set_model_mode
from utils import visualize
from utils import save_model
import torch.optim as optim

# train one epoch
def training_step(net, data_loader, optimizer, cost_function, device):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    # Set the network to training mode
    net.train()

    # Iterate over the training set
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)

        # Loss computation
        loss = cost_function(outputs, targets)
        cumulative_loss += loss.item()

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        samples += inputs.shape[0]
        _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples


def test_step(net, data_loader, cost_function, device):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    # Set the network to evaluation mode
    net.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)

            # Loss computation
            loss = cost_function(outputs, targets)
            cumulative_loss += loss.item()

            # Fetch prediction and loss value
            samples += inputs.shape[0]
            _, predicted = outputs.max(1)

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples


def train(net: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          test_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_folder: str,
          run_name: str):    # -> Dict[str, List]:


    # Computes evaluation results before training
    print("Before training:")
    train_loss, train_accuracy = test_step(net, train_loader, loss_function, device)
    val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)
    test_loss, test_accuracy = test_step(net, test_loader, loss_function, device)

    best_loss = val_loss
    best_model_weights = None
    patience = 3 # at most 3 epoch without improving
    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Log to wandb
    wandb.log({
        "Training accuracy": train_accuracy,
        "Validation accuracy" : val_accuracy
    })

    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    print("-----------------------------------------------------")

    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(net, train_loader, optimizer, loss_function, device)
        scheduler.step()
        val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)

        # Early stopping
        if(val_loss < best_loss):
            best_loss = val_loss
            best_model_weights = copy.deepcopy(net.state_dict())  # Deep copy here      
            patience = 3 # reset patience
        else:
            patience -= 1
            if patience == 0:
                break

        # Log to wandb
        wandb.log({
            "Training accuracy": train_accuracy,
            "Validation accuracy" : val_accuracy
        })

        print(f"Epoch: {e + 1}")
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print("-----------------------------------------------------")


    # Load the best model weights
    save_model(best_model_weights, save_folder, run_name)
    
    # Compute final evaluation results
    print("After training:")
    train_loss, train_accuracy = test_step(net, train_loader, loss_function, device)
    val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)
    test_loss, test_accuracy = test_step(net, test_loader, loss_function, device)

    # Log to wandb
    wandb.log({
        "Training accuracy": train_accuracy,
        "Validation accuracy" : val_accuracy
    })

    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    print("-----------------------------------------------------")


def dann(
    encoder,
    classifier,
    discriminator,
    source_train_loader,
    target_train_loader,
    source_test_loader,
    target_test_loader,
    num_epochs,
    device
):
    print("Training with the DANN adaptation method")

    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_domain = torch.nn.BCELoss()

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=0.001)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    
    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        discriminator.train()

        total_loss = 0
        total_correct = 0

        # Iterate through source and target training data
        for i, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            # Prepare the data
            source_inputs, source_labels = source_data[0].to(device), source_data[1].to(device)
            target_inputs, _ = target_data[0].to(device), target_data[1].to(device)

            # Prepare domain labels
            source_domain_labels = torch.ones(source_inputs.size(0), 1).to(device)
            target_domain_labels = torch.zeros(target_inputs.size(0), 1).to(device)

            # Concatenate inputs for domain classifier
            all_inputs = torch.cat([source_inputs, target_inputs], dim=0)
            all_domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)

            # Zero the gradients
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_discriminator.zero_grad()

            # Forward pass through the encoder
            encoded_features = encoder(all_inputs)

            # Train domain discriminator
            domain_outputs = discriminator(encoded_features.detach())
            domain_loss = criterion_domain(domain_outputs, all_domain_labels)
            domain_loss.backward()
            optimizer_discriminator.step()

            # Forward pass through the classifier
            class_outputs = classifier(encoded_features[:source_inputs.size(0)])

            # Compute classification loss
            class_loss = criterion_class(class_outputs, source_labels)
            class_loss.backward()
            optimizer_encoder.step()
            optimizer_classifier.step()

            # Print statistics
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{min(len(source_train_loader), len(target_train_loader))}], "
                      f"Class Loss: {class_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}")

        # Evaluation phase
        encoder.eval()
        classifier.eval()
        discriminator.eval()

        # Evaluate on source domain
        correct_source, total_source = 0, 0
        with torch.no_grad():
            for source_data in source_test_loader:
                source_inputs, source_labels = source_data[0].to(device), source_data[1].to(device)
                encoded_features = encoder(source_inputs)
                class_outputs = classifier(encoded_features)
                _, predicted = torch.max(class_outputs.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted == source_labels).sum().item()

        # Evaluate on target domain
        correct_target, total_target = 0, 0
        with torch.no_grad():
            for target_data in target_test_loader:
                target_inputs, target_labels = target_data[0].to(device), target_data[1].to(device)
                encoded_features = encoder(target_inputs)
                class_outputs = classifier(encoded_features)
                _, predicted = torch.max(class_outputs.data, 1)
                total_target += target_labels.size(0)
                correct_target += (predicted == target_labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Source Accuracy: {(100 * correct_source / total_source):.2f}%, "
              f"Target Accuracy: {(100 * correct_target / total_target):.2f}%")

    print("Training finished")


# for more reference https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py