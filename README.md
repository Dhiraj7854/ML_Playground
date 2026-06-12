# 🤖 ML Model Playground

An interactive Streamlit app to explore machine learning classifiers on real datasets.  
Pick a dataset, tune hyperparameters, and see results instantly — no code required.

## 🚀 Live Demo
> Deploy to [ML-Playground](https://model-interaction.streamlit.app/) and add your link here.

---

## 📁 Project Structure

```
ml-playground/
│
├── app.py              # Main Streamlit app (UI + layout)
├── models.py           # ML model creation, training, saving & loading
├── utils.py            # Data loading, metrics & chart helpers
├── requirements.txt    # Python dependencies
│
└── saved_models/       # Auto-created — stores trained .pkl files
    ├── KNN_Iris.pkl
    ├── SVM_Wine.pkl
    └── ...
```

### What Each File Does

| File | Role | Analogy |
|------|------|---------|
| `app.py` | UI layout, wires everything together | Waiter |
| `models.py` | Builds, trains, saves, loads models | Chef |
| `utils.py` | Data loading, charts, metrics | Kitchen prep |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Dhiraj7854/ML_Playground.git
cd ML_Playground

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ✨ Features

- **3 Datasets** — Iris, Wine, Breast Cancer (all built-in, no downloads needed)
- **3 Classifiers** — KNN, SVM, Random Forest, Decision Tree
- **Live tuning** — adjust hyperparameters via sliders and see accuracy change
- **Model persistence** — trained models saved as `.pkl` files, auto-reloaded
- **Visualizations** — Confusion Matrix, Feature Importance
- **Model comparison** — run all 4 classifiers and compare accuracy side by side

---

## 🧠 How Model Saving Works

When you click **Train Model**, this happens:

```
User clicks "Train"
    ↓
app.py calls get_model(name, params)       ← models.py
    ↓
app.py calls train_and_save(model, ...)    ← models.py
    ↓
joblib.dump(model, "saved_models/KNN_Iris.pkl")
    ↓
Model stored in st.session_state["model"]
```

On next load (or page refresh):
```
app.py checks st.session_state → not found
    ↓
app.py calls load_model(name, dataset)     ← models.py
    ↓
joblib.load("saved_models/KNN_Iris.pkl")
    ↓
Model ready without retraining!
```

---


## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [scikit-learn](https://scikit-learn.org) — ML models & metrics
- [pandas](https://pandas.pydata.org) — Data handling
- [matplotlib](https://matplotlib.org) / [seaborn](https://seaborn.pydata.org) — Charts
- [joblib](https://joblib.readthedocs.io) — Model serialization