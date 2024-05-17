import torch

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
          device: torch.device):    # -> Dict[str, List]:


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



# for more reference https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py