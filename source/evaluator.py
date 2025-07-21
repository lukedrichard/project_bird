import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, MulticlassAUROC, ConfusionMatrix
import json

def evaluate(device, model, data_loader, results_dir, label_to_index):
    output_dim = model.output_dim
    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)
    precision = Precision(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    recall = Recall(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    f1 = F1Score(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    roc_auc = MulticlassAUROC(num_classes=output_dim, average='weighted').to(device)


    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, labels_onehot = batch  # assuming labels are already numerical
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_onehot = labels_onehot.to(device)

            outputs = model(inputs)

            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)

            _, preds = torch.max(outputs, 1)

            all_preds.append(preds)
            all_labels.append(labels)

    #get metrics

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    accuracy_score = accuracy(all_preds, all_labels).item() * 100
    precision_score = precision(all_preds, all_labels).item()
    recall_score = recall(all_preds, all_labels).item()
    f1_score = f1(all_preds, all_labels).item()
    roc_auc_score = roc_auc(all_probs, all_labels)


    print(f"Post-training evaluation:")
    print(f"Accuracy: {accuracy_score:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Multiclass ROC AUC: {roc_auc_score:.4f}")

    #save metrics in .json file
    metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
    "roc_auc": roc_auc_score.item()
    }

    # Save metrics to JSON file
    with open(results_dir + "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


    #generate confusion matrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=output_dim).to(device)
    cm = confmat(all_preds, all_labels)

    #for labels
    index_to_label = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])] # Reverse mapping from index to label name

    #plot confusion matrix
    plt.figure(figsize=(12, 10))  # Increase size for better readability
    sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
            xticklabels=index_to_label, yticklabels=index_to_label)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better fit
    plt.yticks(rotation=0)
    plt.savefig(results_dir + 'confusion_matrix.png')
    plt.close()
    
    return




def chunk_evaluate(device, model, data_loader, results_dir, label_to_index):
    output_dim = model.output_dim
    #confusion matrix, precision, recall, f1
    accuracy = Accuracy(task='multiclass', num_classes=output_dim, average='micro').to(device)
    precision = Precision(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    recall = Recall(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    f1 = F1Score(task='multiclass',num_classes=output_dim,average='weighted').to(device)
    roc_auc = MulticlassAUROC(num_classes=output_dim, average='weighted').to(device)


    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, label_onehot = batch  # assuming labels are already numerical
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels_onehot = labels_onehot.to(device)

            outputs = model(inputs)

            pooled_outputs = outputs.mean(dim=0)
            log_avg_probs = nn.functional.log_softmax(pooled_outputs, dim=0)

            #probs = nn.functional.softmax(outputs, dim=1)
            #avg_probs = probs.mean(dim=0) 
            pred_class = torch.argmax(log_avg_probs)

            # Step 1: Get predicted class per chunk
            #chunk_preds = torch.argmax(probs, dim=1)
            #votes = torch.bincount(chunk_preds, minlength=len(label_to_index))
            #pred_class = torch.argmax(votes)
            
            all_probs.append(log_avg_probs.unsqueeze(0))
            all_preds.append(pred_class.unsqueeze(0))
            all_labels.append(labels)

    #get metrics

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    print(all_preds)
    print(all_labels)

    accuracy_score = accuracy(all_preds, all_labels).item() * 100
    precision_score = precision(all_preds, all_labels).item()
    recall_score = recall(all_preds, all_labels).item()
    f1_score = f1(all_preds, all_labels).item()
    roc_auc_score = roc_auc(all_probs, all_labels)


    print(f"Post-training evaluation:")
    print(f"Accuracy: {accuracy_score:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Multiclass ROC AUC: {roc_auc_score:.4f}")

    #save metrics in .json file
    metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
    "roc_auc": roc_auc_score.item()
    }

    # Save metrics to JSON file
    with open(results_dir + "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


    #generate confusion matrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=output_dim).to(device)
    cm = confmat(all_preds, all_labels)

    #for labels
    index_to_label = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])] # Reverse mapping from index to label name

    #plot confusion matrix
    plt.figure(figsize=(12, 10))  # Increase size for better readability
    sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
            xticklabels=index_to_label, yticklabels=index_to_label)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better fit
    plt.yticks(rotation=0)
    plt.savefig(results_dir + 'confusion_matrix.png')
    plt.close()
    
    return