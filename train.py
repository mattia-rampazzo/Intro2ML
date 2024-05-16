import torch
#import numpy as np
#import wandb
#import utils
import torch.optim as optim
import torch.nn as nn
#import test
#from tqdm import tqdm
#from utils import save_model
#from utils import visualize
#from utils import set_model_mode



def get_optimizer(net, lr, wd, momentum):
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    return optimizer

def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function

def training_step(net, data_loader, optimizer, cost_function, device):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

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

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def test_step(net, data_loader, cost_function, device):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    # Set the network to evaluation mode
    net.eval()

    # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
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

            # Fetch prediction and loss value
            samples += inputs.shape[0]
            cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100


# Main function
def train(
    net,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    device,
    config,
    learning_rate=0.0001,
    weight_decay=0.0000001,
    momentum=0.9
):

    # Instantiate the optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    # Define the cost function
    loss_function = get_loss_function()

    # Computes evaluation results before training
    print("Before training:")
    train_loss, train_accuracy = test_step(net, train_loader, loss_function, device)
    val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)
    test_loss, test_accuracy = test_step(net, test_loader, loss_function, device)

    # Log to TensorBoard


    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    print("-----------------------------------------------------")

    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(net, train_loader, optimizer, loss_function, device)
        val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)

        # Logs to TensorBoard

        print(f"Epoch: {e + 1}")
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print("-----------------------------------------------------")

    # Compute final evaluation results
    print("After training:")
    train_loss, train_accuracy = test_step(net, train_loader, loss_function, device)
    val_loss, val_accuracy = test_step(net, val_loader, loss_function, device)
    test_loss, test_accuracy = test_step(net, test_loader, loss_function, device)

    # Log to TensorBoard


    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    print("-----------------------------------------------------")