import torch
import math
import wandb
from utils import save_model

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
    all_predictions = []

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

            # Store predictions
            all_predictions.extend(predicted.cpu().numpy())

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples, all_predictions

def train(net: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          test_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_function: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_folder: str,
          run_name: str):    # -> Dict[str, List]:


    # # Computes evaluation results before training
    # print("Before training:")
    # train_loss, train_accuracy, preds = test_step(net, train_loader, loss_function, device)
    # val_loss, val_accuracy, preds = test_step(net, val_loader, loss_function, device)
    # test_loss, test_accuracy, preds = test_step(net, test_loader, loss_function, device)

    # # Log to wandb
    # wandb.log({
    #     "Training loss": train_loss,
    #     "Validation loss" : val_loss,
    #     "Training accuracy": train_accuracy,
    #     "Validation accuracy" : val_accuracy
    # })

    # print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    # print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    # print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    # print("-----------------------------------------------------")

    best_val_loss = math.inf
    best_model_weights = None

    early_stopping_patience = 5 # at most 3 epoch without improving
    epochs_without_improvement = 0 

    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(net, train_loader, optimizer, loss_function, device)
        scheduler.step()
        val_loss, val_accuracy, preds = test_step(net, val_loader, loss_function, device)

        # Log to wandb
        wandb.log({
            "Training loss": train_loss,
            "Validation loss" : val_loss,
            "Training accuracy": train_accuracy,
            "Validation accuracy" : val_accuracy
        })

        print(f"Epoch: {e + 1}")
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print("-----------------------------------------------------")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = net.classifier.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    
        if epochs_without_improvement >= early_stopping_patience:
            print('Early stopping triggered')
            break


    # Load the best model weights
    save_model(best_model_weights, save_folder, run_name)
    
    # # Compute final evaluation results
    # print("After training:")
    # train_loss, train_accuracy, preds = test_step(net, train_loader, loss_function, device)
    # val_loss, val_accuracy, preds = test_step(net, val_loader, loss_function, device)
    # test_loss, test_accuracy, preds = test_step(net, test_loader, loss_function, device)

    # # Log to wandb
    # wandb.log({
    #     "Training loss": train_loss,
    #     "Validation loss" : val_loss,
    #     "Training accuracy": train_accuracy,
    #     "Validation accuracy" : val_accuracy
    # })

    # print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    # print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    # print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    # print("-----------------------------------------------------")



# for more reference https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py