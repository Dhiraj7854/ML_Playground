import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


AVAILABLE_DATASETS = ["Iris", "Wine", "Breast Cancer"]

def load_dataset(name: str):
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Choose from: {AVAILABLE_DATASETS}")

    load = {
        "Iris": datasets.load_iris,
        "Wine": datasets.load_wine,
        "Breast Cancer": datasets.load_breast_cancer
    }

    data = load[name]()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def get_dataset_preview(x, feature_names:list, y,class_names:list) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=feature_names)
    df["Target"] = [class_names[label] for label in y]
    return df

def split_data(x,y,test_size,random_state=42):
    return train_test_split(x,y,random_state=random_state,test_size = test_size)

def get_accuracy(y_test,y_pred)->float:
    return round(accuracy_score(y_test,y_pred) * 100 , 2)

def get_classification_report(y_test,y_pred,class_names:list) -> pd.DataFrame:
    report = classification_report(
        y_test,y_pred,
        target_names = class_names,
        output_dict = True
    )

    df = pd.DataFrame(report).transpose()
    df = df.round(2)
    return df


def plot_confusion_matrix(y_test,y_pred,class_names:list):
    confusion = confusion_matrix(y_test,y_pred)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        confusion,
        annot=True,          
        fmt="d",            
        cmap="Blues",        
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

def plot_feature_importance(model,feature_names:list):        #For RANDOM FOREST
    if not hasattr(model,"feature_importances_"):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
 
    fig, ax = plt.subplots(figsize=(6,5))
    ax.bar(range(len(importances)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_title("Feature Importances (Random Forest)")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    return fig


def plot_accuracy(results:dict):
    fig, ax = plt.subplots(figsize=(6,5))
    models = list(results.keys())
    accuracies = list(results.values())
 
    bars = ax.bar(models, accuracies, color=["#4C72B0", "#DD8452", "#55A868","#C44E52"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
 
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc}%",
            ha="center", va="bottom", fontsize=10
        )
 
    plt.tight_layout()
    return fig
