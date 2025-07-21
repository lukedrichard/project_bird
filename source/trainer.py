import numpy as np
import pandas as pd
import copy
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



def train_model(model, num_epochs, train_loader, val_loader, device, optimizer, criterion, output_dim, results_dir):
    # Cosine Annealing over 100 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1,eta_min=0)

    #train the model
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)

    #early stopping variabels
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        #correct = 0
        #total = 0

        for inputs, targets, targets_onehot in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=1) #convert to log probabilities
            #labels_onehot = torch.zeros_like(log_probs, device=log_probs.device).scatter_(1, targets.unsqueeze(1), 1) #convert labels to 
            loss = criterion(log_probs, targets_onehot)
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
        #val_correct = 0
        #val_total = 0

        accuracy.reset()

        with torch.no_grad():
            for inputs, targets, targets_onehot in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_onehot = targets_onehot.to(device)

                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=1) #convert to log probabilities
                loss = criterion(log_probs, targets_onehot)
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

        # Early stopping logic
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        #update learning rate scheduler
        scheduler.step()

    #Visualize loss and accuracy
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.savefig(results_dir + 'loss_plot.png')  # Save loss plot
    plt.close()  # Close the figure so it doesn't display


    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.savefig(results_dir + 'accuracy_plot.png')  # Save accuracy plot
    plt.close()  # Close the figure

    #reload best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model




def train_chunk_model(model, num_epochs, train_loader, val_loader, device, optimizer, criterion, output_dim, results_dir):
    # Cosine Annealing over 100 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40) #eta_min=0
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1,eta_min=0)

    #train the model
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)

    #early stopping variabels
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        #correct = 0
        #total = 0

        for inputs, targets, targets_onehot in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            #log_probs = F.log_softmax(outputs, dim=1) #convert to log probabilities
            #labels_onehot = torch.zeros_like(log_probs, device=log_probs.device).scatter_(1, targets.unsqueeze(1), 1) #convert labels to 
            
            #loss = criterion(log_probs, targets_onehot) #for single clips

            #for chunks
            pooled_outputs = outputs.mean(dim=0)
            log_avg_probs = F.log_softmax(pooled_outputs, dim=0)
            loss = criterion(log_avg_probs, targets_onehot)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #_, preds = torch.max(outputs.data, 1)
            #accuracy.update(preds, targets)
            #total += targets.size(0)
            #correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        #epoch_acc = 100 * correct / total
        #epoch_acc = accuracy.compute().item() * 100

        train_losses.append(epoch_loss)
        #train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        #val_correct = 0
        #val_total = 0

        accuracy.reset()

        with torch.no_grad():
            for inputs, targets, targets_onehot in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_onehot = targets_onehot.to(device)

                outputs = model(inputs)
                #for chunks
                pooled_outputs = outputs.mean(dim=0)
                log_avg_probs = F.log_softmax(pooled_outputs, dim=0)
                loss = criterion(log_avg_probs, targets_onehot)
                val_loss += loss.item()


                #_, preds = torch.max(outputs.data, 1)
                #accuracy.update(preds, targets)
                #val_total += targets.size(0)
                #val_correct += (predicted == targets).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        #val_epoch_acc = 100 * val_correct / val_total
        #val_epoch_acc = accuracy.compute().item() * 100

        val_losses.append(val_epoch_loss)
        #val_accuracies.append(val_epoch_acc)

        #print(f"Epoch [{epoch+1}/{num_epochs}], "
        #    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
        #    f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}," 
            f"Val Loss: {val_epoch_loss:.4f}")

        # Early stopping logic
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        #update learning rate scheduler
        scheduler.step()

    #Visualize loss and accuracy
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.savefig(results_dir + 'loss_plot.png')  # Save loss plot
    plt.close()  # Close the figure so it doesn't display


    #plt.figure()
    #plt.plot(train_accuracies, label='Train Acc')
    #plt.plot(val_accuracies, label='Val Acc')
    #plt.legend()
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy (%)')
    #plt.title('Accuracy Over Epochs')
    #plt.savefig(results_dir + 'accuracy_plot.png')  # Save accuracy plot
    #plt.close()  # Close the figure

    #reload best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model