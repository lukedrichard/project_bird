import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchaudio #for audio augmentation
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, MulticlassAUROC, ConfusionMatrix

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def train_model(model, num_epochs, train_loader, val_loader, device, optimizer, criterion, output_dim):
    #train the model
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)
    precision = Precision(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    recall = Recall(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    f1 = F1Score(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    roc_auc = MulticlassAUROC(num_classes=output_dim, average='weighted').to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            accuracy.update(preds, targets)
            #total += targets.size(0)
            #correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        #epoch_acc = 100 * correct / total
        epoch_acc = accuracy.compute().item() * 100

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        accuracy.reset()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()


                _, preds = torch.max(outputs.data, 1)
                accuracy.update(preds, targets)
                #val_total += targets.size(0)
                #val_correct += (predicted == targets).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        #val_epoch_acc = 100 * val_correct / val_total
        val_epoch_acc = accuracy.compute().item() * 100

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


    #Visualize loss and accuracy
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.savefig('loss_plot.png')  # Save loss plot
    plt.close()  # Close the figure so it doesn't display


    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.savefig('accuracy_plot.png')  # Save accuracy plot
    plt.close()  # Close the figure