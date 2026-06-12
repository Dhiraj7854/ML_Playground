import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

MODELS_DIR = "saved models"

os.makedirs(MODELS_DIR,exist_ok=True)

def get_model(classifier_name:str,parameters:dict):
    if classifier_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=parameters.get("n_neighbors",5))
    
    elif classifier_name == 'SVM':
        return SVC(
            C=parameters.get('C',1.0),
            kernel=parameters.get('kernel','rbf'),
            probability=True
        )
    elif classifier_name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=parameters.get("n_estimators",100),
            max_depth=parameters.get("max_depth",None),
            criterion=parameters.get("criterion","gini"),
            random_state=42
        )
    elif classifier_name == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=parameters.get("max_depth",None),
            min_samples_leaf=parameters.get("min_samples_leaf",1),
            min_samples_split=parameters.get("min_samples_split",2),
            criterion=parameters.get("criterion","gini"),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown Classifier: {classifier_name}")


def train_and_save(model,X_train,Y_train,classifier_name:str,dataset_name:str):
    model.fit(X_train,Y_train)

    filename = f"{classifier_name.replace(' ', '')}_{dataset_name}.pkl"
    save_path = os.path.join(MODELS_DIR,filename)

    joblib.dump(model,save_path)
    print(f"[models.py] Model saved to: {save_path}")

    return model


def load_model(classifier_name:str,dataset_name:str):
    filename = f"{classifier_name.replace(' ', '')}_{dataset_name}.pkl"
    load_path = os.path.join(MODELS_DIR,filename)

    if os.path.exists(load_path):
        model = joblib.load(load_path)
        print(f"[model.py] loaded from: {load_path}")
        return model
    return None


def get_hyperparameter_config(classifier_name:str) -> dict:
    configs = {
        "KNN": {
            "n_neighbors": {
                "type": "slider",
                "label": "Number of Neighbors (K)",
                "min": 1, "max":20, "default":5, "step": 1
            }
        },

        "SVM": {
            "C": {
                "type": "slider",
                "label": "Regularization (C)",
                "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01
            },

            "kernel": {
                "type": "selectbox",
                "label": "Kernel",
                "options": ["rbf","linear","poly"],
                "default": "rbf"
            }
        },

        "Random Forest": {
            "n_estimators": {
                "type": "slider",
                "label": "Number of Trees",
                "min": 10, "max": 300, "default": 100, "step": 1
            },

            "max_depth": {
                "type": "slider",
                "label": "Max Tree Depth(0 = Unlimited)",
                "min": 0, "max": 20, "default": 0, "step": 1
            },

            "criterion": {
                "type": "selectbox",
                "label": "Quality of Split",
                "options": ["gini","entropy","log_loss"],
                "default": "gini"
            }
        },

        "Decision Tree": {
            "max_depth": {
                "type": "slider",
                "label": "Max Tree Depth(0 = Unlimited)",
                "min": 0, "max": 20, "default": 0, "step": 1
            },

            "min_samples_leaf": {
                "type": "slider",
                "label": "Minimum Samples Required in Leaf",
                "min": 1, "max": 20, "default": 1, "step": 1
            },

            "min_samples_split": {
                "type":"slider",
                "label": "Minimum Samples in Internal Nodes",
                "min": 2, "max": 20, "default": 2, "step":1
            },

            "criterion": {
                "type": "selectbox",
                "label": "Quality of Split",
                "options": ["gini","entropy","log_loss"],
                "default": "gini"
            }
        }
    }

    return configs.get(classifier_name,{})