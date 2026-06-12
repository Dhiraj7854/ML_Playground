import streamlit as st
import pandas as pd

from models import get_model, train_and_save, load_model, get_hyperparameter_config

from utils import (
    AVAILABLE_DATASETS,
    load_dataset,
    get_dataset_preview,
    split_data,
    get_accuracy,
    get_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_accuracy
)

st.set_page_config(page_title="ML Playground", page_icon="🤖", layout="centered")

st.sidebar.title("CONTROLS")
st.sidebar.markdown("---")

# Dataset
st.sidebar.subheader("DATASET")
dataset_name = st.sidebar.selectbox("Choose a dataset", AVAILABLE_DATASETS)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
test_size_fraction = test_size / 100

st.sidebar.markdown("---")

# Classifier
st.sidebar.subheader("CLASSIFIER")
CLASSIFIERS = ["KNN", "SVM", "Random Forest", "Decision Tree"]
classifier_name = st.sidebar.selectbox("Choose a classifier", CLASSIFIERS)

st.sidebar.subheader("HYPERPARAMETERS")
param_config = get_hyperparameter_config(classifier_name)
user_params = {}

for param_name, config in param_config.items():
    if config["type"] == "slider":
        value = st.sidebar.slider(
            config["label"],
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"],
            step=config["step"]
        )
        user_params[param_name] = None if (param_name == "max_depth" and value == 0) else value

    elif config["type"] == "selectbox":
        value = st.sidebar.selectbox(config["label"], config["options"])
        user_params[param_name] = value

st.sidebar.markdown("---")

# Action
train_button = st.sidebar.button("Train Model")
compare_button = st.sidebar.button("Compare All Models")

st.title("🤖 ML Model Playground")
st.markdown("Explore machine learning classifiers with real datasets. Tune hyperparameters and see results instantly.")
st.markdown("---")


@st.cache_data
def get_data(name):
    return load_dataset(name)

X, y, feature_names, class_names = get_data(dataset_name)
X_train, X_test, y_train, y_test = split_data(X, y, test_size_fraction)

# Dataset info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", dataset_name)
col2.metric("Total Samples", len(X))
col3.metric("Features", X.shape[1])
col4.metric("Classes", len(class_names))

# Dataset preview
with st.expander("Preview Dataset (first 10 rows)"):
    df_preview = get_dataset_preview(X, feature_names, y, class_names)
    st.dataframe(df_preview.head(10), width='stretch')


if train_button:
    with st.spinner(f"Training {classifier_name} on {dataset_name}..."):

        model = get_model(classifier_name, user_params)

        model = train_and_save(model, X_train, y_train, classifier_name, dataset_name)

        st.session_state["model"] = model
        st.session_state["trained_classifier"] = classifier_name
        st.session_state["trained_dataset"] = dataset_name

    st.success(f"{classifier_name} trained successfully and saved to `saved_models/`!")


model = st.session_state.get("model", None)

if model is None:
    model = load_model(classifier_name, dataset_name)
    if model is not None:
        st.session_state["model"] = model
        st.info(f"Loaded previously saved **{classifier_name}** model for **{dataset_name}**.")

if model is not None:
    y_pred = model.predict(X_test)
    accuracy = get_accuracy(y_test, y_pred)

    # Results
    st.subheader(f"Results — {classifier_name} on {dataset_name}")

    # Accuracy 
    col_acc, col_train, col_test = st.columns(3)
    col_acc.metric("🎯 Accuracy", f"{accuracy}%")
    col_train.metric("🔵 Training Samples", len(X_train))
    col_test.metric("🟠 Test Samples", len(X_test))

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🟦 Confusion Matrix")
        st.markdown("Diagonal = correct predictions. Off-diagonal = errors.")
        fig_cm = plot_confusion_matrix(y_test, y_pred, class_names)
        st.pyplot(fig_cm,width='stretch')

    with col_right:
        if classifier_name == "Random Forest":
            st.markdown("#### Feature Importance")
            st.markdown("Which features the model relies on most.")
            fig_fi = plot_feature_importance(model, feature_names)
            if fig_fi:
                st.pyplot(fig_fi,width='stretch')
        else:
            st.markdown("#### 📋 Classification Report")
            st.markdown("Precision, recall, and F1-score per class.")
            report_df = get_classification_report(y_test, y_pred, class_names)
            st.dataframe(report_df, width='stretch')

    # Classification report
    with st.expander("📋 Full Classification Report"):
        report_df = get_classification_report(y_test, y_pred, class_names)
        st.dataframe(report_df, width='stretch')

else:
    st.info("👈 Choose a dataset and classifier in the sidebar, then click **Train Model** to begin.")


if compare_button:
    st.markdown("---")
    st.subheader("Model Comparison — All Classifiers")
    st.markdown(f"Running all 4 classifiers on **{dataset_name}** with default hyperparameters...")

    comparison_results = {}
    progress = st.progress(0)

    for i, clf_name in enumerate(CLASSIFIERS):
        default_params = {
            "n_neighbors": 5,
            "C": 1.0,
            "kernel": "rbf",
            "n_estimators": 100,
            "max_depth": None
        }
        clf = get_model(clf_name, default_params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        comparison_results[clf_name] = get_accuracy(y_test, preds)
        progress.progress((i + 1) / len(CLASSIFIERS))

    fig_compare = plot_accuracy(comparison_results)
    st.pyplot(fig_compare)

    compare_df = pd.DataFrame(
        list(comparison_results.items()),
        columns=["Classifier", "Accuracy (%)"]
    ).sort_values("Accuracy (%)", ascending=False)
    st.dataframe(compare_df, width='stretch')


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>ML Playground · Built with Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)