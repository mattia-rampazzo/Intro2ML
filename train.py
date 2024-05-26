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
        for (source_data, source_labels), (target_data, _) in zip(source_train_loader, target_train_loader):
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            
            # Domain labels
            source_domain_labels = torch.ones(source_data.size(0), 1).to(device)
            target_domain_labels = torch.zeros(target_data.size(0), 1).to(device)

            # Zero gradients
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_discriminator.zero_grad()

            # Forward pass
            source_features = encoder(source_data)
            target_features = encoder(target_data)

            source_class_preds = classifier(source_features)
            source_domain_preds = discriminator(source_features)
            target_domain_preds = discriminator(target_features)

            # Calculate losses
            class_loss = criterion_class(source_class_preds, source_labels)
            domain_loss = criterion_domain(source_domain_preds, source_domain_labels) + \
                          criterion_domain(target_domain_preds, target_domain_labels)
            
            # Backpropagation and optimization
            class_loss.backward(retain_graph=True)
            optimizer_classifier.step()
            
            domain_loss.backward()
            optimizer_encoder.step()
            optimizer_discriminator.step()

            # Update total loss and correct predictions
            total_loss += class_loss.item()
            _, predicted = torch.max(source_class_preds.data, 1)
            total_correct += (predicted == source_labels).sum().item()

        # Calculate and print epoch loss and accuracy
        epoch_loss = total_loss / len(source_train_loader)
        epoch_accuracy = total_correct / len(source_train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Evaluation on test data
        test.evaluate(encoder, classifier, source_test_loader, target_test_loader, device)



# for more reference https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py