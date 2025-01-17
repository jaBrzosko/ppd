import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SAVE_PATH = "./summaries"

def plot_confusion_matrix(cm, labels):

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def model_summary(y_true, y_pred, label_names, model_name, is_y_indices=False):
    if is_y_indices:
        y_true, y_pred = np.take(label_names, y_true), np.take(label_names, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    return {"name": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "confusion_matrix": cm, "labels": label_names}

def save_model_summary_dict(summary_dict):
    model_name = summary_dict["name"]
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, f"{model_name}.pickle"), "wb+") as f:
        pickle.dump(summary_dict, f)

def load_model_summary_dicts():
    summaries = []
    for model_name in os.listdir(SAVE_PATH):
        model_path = os.path.join(SAVE_PATH, model_name)
        if os.path.isdir(model_path):
            continue

        with open(model_path, "rb") as f:
            summary_dict = pickle.load(f)
            summaries.append(summary_dict)

    return summaries

def show_model_summary_dict(summary_dict):
    summary_dict = summary_dict.copy()
    model_name = summary_dict.pop("name")
    cm = summary_dict.pop("confusion_matrix")
    labels = summary_dict.pop("labels")

    for key, value in summary_dict.items():
        print(f"{model_name} {key}: {value:.2f}")

    plot_confusion_matrix(cm, labels)