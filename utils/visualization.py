import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history, save_dir=None):
    """
    Plots the training and validation accuracy and loss.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Plot Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir=None):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()

def generate_classification_report(y_true, y_pred, classes, save_dir=None):
    """
    Generates and optionally saves the classification report.
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n")
    print(report)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
            
def plot_class_distribution(train_dir, test_dir):
    """
    Plots the distribution of classes in train and test datasets.
    """
    def get_counts(directory):
        classes = sorted(os.listdir(directory))
        counts = [len(os.listdir(os.path.join(directory, c))) for c in classes]
        return classes, counts

    train_classes, train_counts = get_counts(train_dir)
    test_classes, test_counts = get_counts(test_dir)
    
    df_train = pd.DataFrame({'Class': train_classes, 'Count': train_counts, 'Set': 'Train'})
    df_test = pd.DataFrame({'Class': test_classes, 'Count': test_counts, 'Set': 'Test'})
    df = pd.concat([df_train, df_test])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', hue='Set', data=df)
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.show()
