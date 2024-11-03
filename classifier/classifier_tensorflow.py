import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import tensorflow_decision_forests as tfdf
#from plotting.config import get_mapping

"""
Utility configs
"""

MAPPING_FEMALES = {
    "WMS_Baseline": "White Matter Stroke (Baseline, Female)",
    "WMS_7dpi": "White Matter Stroke (7 dpi, Female)",
    "WMS_30dpi": "White Matter Stroke (30 dpi, Female)",
    "CS_Baseline": "Cortical Stroke (Baseline, Female)",
    "CS_7dpi": "Cortical Stroke (7 dpi, Female)",
    "CS_30dpi": "Cortical Stroke (30 dpi, Female)",
    "Control": "Control (Baseline, Female)",
    "Control_7dpi": "Control (7 days, Female)",
    "Control_30dpi": "Control (30 days, Female)",
}

COLOR_MAPPING_FEMALES = {
    "White Matter Stroke (Baseline, Female)": "#1f77b4",
    "White Matter Stroke (7 dpi, Female)": "#ff7f0e",
    "White Matter Stroke (30 dpi, Female)": "#4c72b0",
    "Cortical Stroke (Baseline, Female)": "#dd8452",
    "Cortical Stroke (7 dpi, Female)": "#a1c9f4",
    "Cortical Stroke (30 dpi, Female)": "#ffb482",
    "Control (Baseline, Female)": "#4878d0",
    "Control (7 days, Female)": "#ee854a",
    "Control (30 days, Female)": "#001c7f",
}

MAPPING_MALES = {
    "WMS_Baseline": "White Matter Stroke (Baseline, Male)",
    "WMS_7dpi": "White Matter Stroke (7 dpi, Male)",
    "WMS_30dpi": "White Matter Stroke (30 dpi, Male)",
    "CS_Baseline": "Cortical Stroke (Baseline, Male)",
    "CS_7dpi": "Cortical Stroke (7 dpi, Male)",
    "CS_30dpi": "Cortical Stroke (30 dpi, Male)",
    "Control_bsl": "Control (Baseline, Male)",
    "Control_7dpi": "Control (7 days, Male)",
    "Control_30dpi": "Control (30 days, Male)",
}

COLOR_MAPPING_MALES = {
    "White Matter Stroke (Baseline, Male)": "#1f77b4",
    "White Matter Stroke (7 dpi, Male)": "#ff7f0e",
    "White Matter Stroke (30 dpi, Male)": "#4c72b0",
    "Cortical Stroke (Baseline, Male)": "#dd8452",
    "Cortical Stroke (7 dpi, Male)": "#a1c9f4",
    "Cortical Stroke (30 dpi, Male)": "#ffb482",
    "Control (Baseline, Male)": "#e377c2",
    "Control (7 days, Male)": "#7f7f7f",
    "Control (30 days, Male)": "#bcbd22",
}


def get_mapping(mapping):
    """
    Returns mappings as a tuple of (mapping, color_mapping)
    """
    if mapping.lower() == "male":
        return MAPPING_MALES, COLOR_MAPPING_MALES
    elif mapping.lower() == "female":
        return MAPPING_FEMALES, COLOR_MAPPING_FEMALES
    else:
        raise ValueError(f"Mapping must be male or female, not {mapping}")

class Classifier:
    """
    TensorFlow Classifier for predicting experimental group based on behavioral summaries.
    Incorporates SMOTE for handling class imbalance.
    """

    def __init__(
        self,
        model_type="logistic_regression",
        color_mapping=None,
        random_state=42,
        plot_dir="plots/model_plots",
        num_epochs=50,
        features="moseq",
        num_classes=None,
        input_dim=None,
        label_encoder=None,
        num_trees=100,
        max_depth=16,
        batch_size=32,
        verbose=0,
    ):
        """
        Initialize Classifier.

        Args:
        - model_type (str): Type of model to use ('logistic_regression', 'deep_nn', 'random_forest', 'svm').
        - color_mapping (dict): Dictionary mapping class labels to colors.
        - random_state (int): Random state used for reproducibility.
        - plot_dir (str): Path to save plots in.
        - num_epochs (int): Number of epochs to train the model for.
        - features (str): Prediction features being used.
        - num_classes (int): Number of output classes.
        - input_dim (int): Number of input features.
        - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
        - num_trees (int): Number of Decision Trees for the Random Forest model.
        - max_depth (int): The maximum number of nodes in each trees.
        - batch_size (int): The number of samples used in one forward and backwards pass through the network.
        - verbose (int): Verbosity mode for logging.
        
        Attributes:
        - model_type (str): Type of model to use.
        - color_mapping (dict): Color mapping for classes.
        - random_state (int): Random state used for reproducibility.
        - plot_dir (str): Path to save plots in.
        - num_epochs (int): Number of epochs to train the model for.
        - features (str): Prediction features being used.
        - num_classes (int): Number of output classes.
        - input_dim (int): Number of input features.
        - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
        - num_trees (int): Number of Decision Trees for the Random Forest model.
        - max_depth (int): The maximum number of nodes in each trees.
        - batch_size (int): The number of samples used in one forward and backwards pass through the network.
        - verbose (int): Verbosity mode for logging.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.plot_dir = plot_dir
        self.features = features
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.label_encoder = label_encoder
        self.num_epochs = num_epochs

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.skf = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=self.random_state
        )
        
        self.color_mapping = get_mapping(color_mapping) if color_mapping else {}

        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=self.random_state)

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.verbose = verbose

    def _get_model(self):
        """
        Builds and returns a TensorFlow model based on the specified `model_type`.
        """
        if self.model_type == "logistic_regression":
            model = Sequential([
                Input(shape=(self.input_dim,)),
                Dense(self.num_classes, activation="softmax")
            ])
            model.compile(
                optimizer=Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        elif self.model_type == "deep_nn":
            model = Sequential([
                Input(shape=(self.input_dim,)),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(self.num_classes, activation="softmax")
            ])
            model.compile(
                optimizer=Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        elif self.model_type == "random_forest":
            return None  # Random Forest model handled separately
        elif self.model_type == "svm":
            model = Sequential([
                Input(shape=(self.input_dim,)),
                Dense(self.num_classes, activation="linear")
            ])
            model.compile(
                optimizer=Adam(),
                loss="hinge",
                metrics=["accuracy"],
            )
            return model
        else:
            raise ValueError(
                "Invalid model type. Choose 'logistic_regression', 'deep_nn', 'random_forest', or 'svm'."
            )

    def _preprocess_data(self, X, y):
        """
        Scales the data with StandardScaler and applies SMOTE to handle class imbalance.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        X_resampled (numpy.array): Resampled and scaled features.
        y_resampled (numpy.array): Resampled labels.
        """
        X_scaled = self.scaler.fit_transform(X)
        X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
        return X_resampled, y_resampled

    def cross_validate(self, X, y):
        """
        Perform cross-validation, train models, and collect predictions, true labels, and probabilities.

        Returns:
        - y_true: numpy array of true labels
        - y_pred: numpy array of predicted labels
        - y_scores: numpy array of predicted probabilities (if applicable)
        """
        y_true = []
        y_pred = []
        y_scores = []

        X_resampled, y_resampled = self._preprocess_data(X, y)

        for train_index, test_index in self.skf.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            if self.model_type == "random_forest":
                # For TFDF, convert data to TensorFlow datasets
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                # Create the model
                model = tfdf.keras.RandomForestModel(
                    random_seed=self.random_state,
                    num_trees=self.num_trees,
                    max_depth=self.max_depth,
                )
                model.compile(metrics=["accuracy"])
                model.fit(train_ds)

                y_prob = model.predict(test_ds)
                y_pred_labels = np.argmax(y_prob, axis=1)
            else:
                # Convert labels to categorical
                y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
                y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

                # Compute class weights
                class_weights = class_weight.compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(y_train),
                    y=y_train,
                )
                class_weights = dict(enumerate(class_weights))

                model = self._get_model()

                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )

                model.fit(
                    X_train,
                    y_train_cat,
                    epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_test, y_test_cat),
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=self.verbose,
                )

                y_prob = model.predict(X_test)
                y_pred_labels = np.argmax(y_prob, axis=1)

            y_true.extend(y_test)
            y_pred.extend(y_pred_labels)
            y_scores.extend(y_prob)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        return y_true, y_pred, y_scores

    def train_and_evaluate(self, X, y):
        """
        Trains, fits, and evaluates the model using cross-validation.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        tuple: (y_pred, accuracy, conf_matrix, class_report, f1, cm_fig)
        """
        y_true, y_pred, _ = self.cross_validate(X, y)

        accuracy = np.mean(y_pred == y_true)
        conf_matrix = confusion_matrix(y_true, y_pred)

        if self.label_encoder is not None:
            y_true_labels = self.label_encoder.inverse_transform(y_true)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            class_names = self.label_encoder.classes_
        else:
            y_true_labels = y_true
            y_pred_labels = y_pred
            class_names = np.unique(y_true)

        class_report = classification_report(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")

        def plot_cm(conf_matrix, class_names):
            fig = plt.figure(figsize=(8, 6), dpi=200)
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 12},
                cbar_kws={"shrink": 0.75},
            )
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.yticks(fontsize=7)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)
            plt.title("Confusion Matrix", fontsize=14)
            plt.tight_layout()

            fig_name = f"{self.features}_confusion_matrix_{self.model_type}.png"
            plt.savefig(os.path.join(self.plot_dir, fig_name))
            return fig

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(class_report)
        print(f"F1 Score: {f1:.4f}")

        cm_fig = plot_cm(conf_matrix, class_names)

        return y_pred, accuracy, conf_matrix, class_report, f1, cm_fig

    def get_feature_importance(self, X, y):
        """
        Calculates and returns feature importance for logistic regression, linear SVM, and Random Forest models.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series or numpy.array): Target labels.

        Returns:
        importances_df (pandas.DataFrame): A DataFrame containing features and their importance scores.
        """
        if self.model_type not in ["logistic_regression", "svm", "random_forest"]:
            raise ValueError(
                "Feature importance is only available for logistic regression, linear SVM, and Random Forest models."
            )

        X_resampled, y_resampled = self._preprocess_data(X, y)

        if self.model_type == "random_forest":
            # Convert data to TensorFlow dataset
            train_ds = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))

            # Create and train the model
            model = tfdf.keras.RandomForestModel(
                random_seed=self.random_state,
                num_trees=self.num_trees,
                max_depth=self.max_depth,
            )
            model.compile(metrics=["accuracy"])
            model.fit(train_ds)

            inspector = model.make_inspector()
            importances = inspector.variable_importances()["NUM_NODES"]

            # Extract feature names and importance scores
            feature_names = X.columns.tolist()
            importance_dict = {imp.feature: imp.importance for imp in importances}
            importances_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": [importance_dict.get(name, 0) for name in feature_names],
            })
            importances_df = importances_df.sort_values(by="Importance", ascending=False)
        else:
            # For logistic regression and SVM
            # Initialize weights accumulator
            feature_weights = np.zeros(self.input_dim)

            # Cross-validation to accumulate weights
            for train_index, _ in self.skf.split(X_resampled, y_resampled):
                X_train, y_train = X_resampled[train_index], y_resampled[train_index]

                # Convert labels to categorical
                y_train_cat = to_categorical(y_train, num_classes=self.num_classes)

                # Compute class weights
                class_weights = class_weight.compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(y_train),
                    y=y_train,
                )
                class_weights = dict(enumerate(class_weights))

                model = self._get_model()

                # Train the model
                model.fit(
                    X_train,
                    y_train_cat,
                    epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    class_weight=class_weights,
                    verbose=self.verbose,
                )

                # Extract weights from the model
                weights = model.get_weights()[0]  # Get the weights of the first layer

                # For multi-class, average the absolute weights across classes
                if self.num_classes > 2:
                    mean_abs_weights = np.mean(np.abs(weights), axis=1)
                else:
                    mean_abs_weights = np.abs(weights).flatten()

                feature_weights += mean_abs_weights

            # Average the accumulated weights
            feature_weights /= self.skf.get_n_splits()

            # Create a DataFrame with feature names and their importance scores
            feature_names = X.columns.tolist()
            importances_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": feature_weights,
            })
            importances_df = importances_df.sort_values(by="Importance", ascending=False)

        print("Top 10 Feature Importances:")
        print(importances_df.head(10))

        save_path = os.path.join(self.plot_dir, f"feature_importances_{self.model_type}.csv")
        importances_df.to_csv(save_path, index=False)

        return importances_df

    def plot_precision_recall_curve(self, X, y):
        """
        Plots the Precision-Recall curve for each class.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        fig (matplotlib.figure.Figure): The generated Precision-Recall curve plot.
        """
        y_true, _, y_scores = self.cross_validate(X, y)

        # Binarize the output
        y_test_binarized = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_test_binarized.shape[1]

        # Compute PR curve and average precision for each class
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_binarized[:, i], y_scores[:, i]
            )
            average_precision[i] = average_precision_score(
                y_test_binarized[:, i], y_scores[:, i]
            )

        # Plot the PR curve
        fig = plt.figure(figsize=(6, 5), dpi=200)

        for i, class_label in enumerate(np.unique(y_true)):
            if self.label_encoder is not None:
                class_name = self.label_encoder.inverse_transform([class_label])[0]
            else:
                class_name = str(class_label)
            color = self.color_mapping.get(class_label, plt.cm.jet(float(i) / n_classes))
            plt.plot(
                recall[i],
                precision[i],
                color=color,
                lw=2,
                label=f"Class {class_name} (AP = {average_precision[i]:0.2f})",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f'Precision-Recall Curve - {self.model_type.replace("_", " ").title()}'
        )
        plt.legend(loc="lower left", fontsize=8)
        sns.set_style("whitegrid")

        # Save the figure
        fig_name = f"{self.features}_pr_curve_{self.model_type}.png"
        plt.savefig(os.path.join(self.plot_dir, fig_name))
        plt.tight_layout()

        return fig
