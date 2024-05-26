import torch
import wandb
import numpy as np
from model import DomainClassifier
from utils import set_model_mode


def evaluate(encoder, classifier, source_test_loader, target_test_loader, device):
    encoder.eval()
    classifier.eval()

    source_correct = 0
    target_correct = 0
    source_total = 0
    target_total = 0

    with torch.no_grad():
        for data, labels in source_test_loader:
            data, labels = data.to(device), labels.to(device)
            features = encoder(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            source_total += labels.size(0)
            source_correct += (predicted == labels).sum().item()

        for data, labels in target_test_loader:
            data, labels = data.to(device), labels.to(device)
            features = encoder(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            target_total += labels.size(0)
            target_correct += (predicted == labels).sum().item()

    source_accuracy = source_correct / source_total
    target_accuracy = target_correct / target_total

    print(f'Source Test Accuracy: {source_accuracy:.4f}, Target Test Accuracy: {target_accuracy:.4f}')
