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
    device,
    config,
):
    print("Training with the DANN adaptation method")

    classifier_criterion = nn.CrossEntropyLoss().to(device)
    discriminator_criterion = nn.BCELoss().to(device)

    optimizer = optim.SGD(
        list(encoder.parameters())
        + list(classifier.parameters())
        + list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9,
    )
    
    
    for epoch in tqdm(range(num_epochs)):
        set_model_mode("train", [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = num_epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(
            zip(source_train_loader, target_train_loader)
        ):
            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.to(device), source_label.to(
                device
            )
            target_image, target_label = target_image.to(device), target_label.to(
                device
            )
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(
                torch.LongTensor
            )
            domain_target_labels = torch.ones(target_label.shape[0]).type(
                torch.LongTensor
            )
            domain_combined_label = torch.cat(
                (domain_source_labels, domain_target_labels), 0
            ).to(device).unsqueeze(1).float()
            
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()
            
            
            wandb.log({
                "DANN/loss": total_loss.item()},
            )

        test.tester(
            encoder,
            classifier,
            discriminator,
            source_test_loader,
            target_test_loader,
            training_mode="DANN",
            device=device,
            epoch=epoch,
        )

    save_model(encoder, classifier, discriminator, "DANN")
    if config['training']['visualize']:
        visualize(encoder, 'DANN', source_test_loader, target_test_loader)

# for more reference https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py