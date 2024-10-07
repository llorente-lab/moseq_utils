import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    f1_score,
    average_precision_score,
)
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import tensorflow_decision_forests as tfdf

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from plotting.config import _get_mapping

class Classifier:
    """
    TensorFlow-based Classifier for predicting experimental group based on behavioral summaries.
    Incorporates SMOTE for handling class imbalance. The Scikit-Learn implementation here is replaced with a Tensorflow-based implementation, including the addition of a deep neural network classifier.
    """

    def __init__(
        self,
        model_type="logistic_regression",
        color_mapping=None,
        random_state=42,
        plot_dir="plots/model_plots",
        features="moseq",
        num_classes=None,
        input_dim=None,
    ):
        """
        Initialize Classifier.

        Args:
        - model_type (str): Type of model to use ('logistic_regression', 'deep_nn', 'random_forest', 'svm').
        - color_mapping (str): String which specifies which color mapping to use.
        - random_state (int): Random state used for reproducibility.
        - plot_dir (str): Path to save plots in.
        - features (str): Prediction features being used.
        - num_classes (int): Number of output classes.
        - input_dim (int): Number of input features.

        Attributes:
        - model_type (str): Type of model to use.
        - color_mapping (dict): Dictionary with keys corresponding to experimental group and values corresponding to colors.
        - random_state (int): Random state used for reproducibility.
        - plot_dir (str): Path to save plots in.
        - model: The TensorFlow model instantiated based on the specified `model_type`.
        - scaler (sklearn.preprocessing.StandardScaler): Scaler used to standardize features.
        - smote (SMOTE): SMOTE instance for handling class imbalance.
        - skf (StratifiedKFold): K-fold cross-validation splitter with stratification.
        - features (str): Prediction features being used.
        - num_classes (int): Number of output classes.
        - input_dim (int): Number of input features.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.plot_dir = plot_dir
        self.features = features
        self.num_classes = num_classes
        self.input_dim = input_dim

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.skf = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=self.random_state
        )

        if color_mapping is not None:
            _, self.color_mapping = _get_mapping(color_mapping)
        else:
            self.color_mapping = {}

        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=self.random_state)
    def _get_model(self):
      """
      Builds and returns a TensorFlow model based on the specified `model_type`.
      """
      if self.model_type == "logistic_regression":
          model = Sequential()
          model.add(Input(shape=(self.input_dim,)))  # Define input shape explicitly
          model.add(
              Dense(
                  self.num_classes,
                  activation="softmax",
              )
          )
          model.compile(
              optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"],
          )
          return model
      elif self.model_type == "deep_nn":
          model = Sequential()
          model.add(Input(shape=(self.input_dim,)))  # define input shape here explicitly or else we run into errors
          model.add(Dense(128, activation="relu"))
          model.add(Dropout(0.5))
          model.add(Dense(64, activation="relu"))
          model.add(Dropout(0.5))
          model.add(Dense(self.num_classes, activation="softmax"))
          model.compile(
              optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"],
          )
          return model
      elif self.model_type == "random_forest":
          # For TensorFlow Decision Forests, we don't need to define a model here, and can do so later
          return None
      elif self.model_type == "svm":
          model = Sequential()
          model.add(Input(shape=(self.input_dim,)))  
          model.add(
              Dense(
                  self.num_classes,
                  activation="linear",
              )
          )
          model.compile(
              optimizer=Adam(),
              loss="hinge",  # Using hinge loss for SVM
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

    def train_and_evaluate(self, X, y):
        """
        Trains, fits, and evaluates the model using cross-validation.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        tuple: (y_pred, accuracy, conf_matrix, class_report, f1, cm_fig)
        """
        X_resampled, y_resampled = self._preprocess_data(X, y)

        y_true = []
        y_pred = []

        for train_index, test_index in self.skf.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            if self.model_type == "random_forest":
                # For TFDF, we need to convert data to TensorFlow datasets
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                # Create the model
                model = tfdf.keras.RandomForestModel(
                    random_seed=self.random_state,
                    num_trees=100,
                    max_depth=16,
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

                self.model = self._get_model()

                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )

                self.model.fit(
                    X_train,
                    y_train_cat,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test_cat),
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=0,
                )

                y_prob = self.model.predict(X_test)
                y_pred_labels = np.argmax(y_prob, axis=1)

            y_true.extend(y_test)
            y_pred.extend(y_pred_labels)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_pred == y_true)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

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

        class_names = np.unique(y_true)
        cm_fig = plot_cm(conf_matrix, class_names)
        print(f"F1 Score: {f1:.4f}")

        return y_pred, accuracy, conf_matrix, class_report, f1, cm_fig

    def get_feature_importance(self, X, y):
        """
        Calculates and returns feature importance for the Random Forest model.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        importances_df (pandas.DataFrame): A DataFrame containing features and their importance scores.
        """
        if self.model_type != "random_forest":
            raise ValueError("Feature importance is only available for Random Forest model.")

        X_resampled, y_resampled = self._preprocess_data(X, y)

        # Convert data to TensorFlow dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))

        # Create and train the model
        model = tfdf.keras.RandomForestModel(
            random_seed=self.random_state,
            num_trees=100,
            max_depth=16,
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

        print("Top 10 Feature Importances:")
        print(importances_df.head(10))

        save_path = os.path.join(self.plot_dir, "feature_importances.csv")
        importances_df.to_csv(save_path, index=False)

        return importances_df

    def plot_precision_recall_curve(self, X, y):
        """
        Plots the Precision-Recall curve for each class and the micro-average.

        Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target labels.

        Returns:
        matplotlib.figure.Figure: The generated Precision-Recall curve plot.
        """
        X_resampled, y_resampled = self._preprocess_data(X, y)

        y_true = []
        y_scores = []

        for train_index, test_index in self.skf.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            if self.model_type == "random_forest":
                # For TFDF, we need to convert data to TensorFlow datasets
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                # Create the model
                model = tfdf.keras.RandomForestModel(
                    random_seed=self.random_state,
                    num_trees=100,
                    max_depth=16,
                )
                model.compile(metrics=["accuracy"])
                model.fit(train_ds)

                y_prob = model.predict(test_ds)
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

                self.model = self._get_model()

                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )

                self.model.fit(
                    X_train,
                    y_train_cat,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test_cat),
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=0,
                )

                y_prob = self.model.predict(X_test)

            y_true.extend(y_test)
            y_scores.extend(y_prob)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

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

        # Compute micro-average PR curve and AP
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_binarized.ravel(), y_scores.ravel()
        )
        average_precision["micro"] = average_precision_score(
            y_test_binarized, y_scores, average="micro"
        )

        # Plot the PR curve
        fig = plt.figure(figsize=(6, 5), dpi=200)
        lines, labels = [], []

        # Plot PR curve for each class
        for i, class_label in enumerate(np.unique(y_true)):
            if class_label in self.color_mapping:
                color = self.color_mapping[class_label]
            else:
                color = plt.cm.jet(float(i) / n_classes)
            plt.plot(
                recall[i],
                precision[i],
                color=color,
                lw=2,
                label=f"Class {class_label} (AP = {average_precision[i]:0.2f})",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f'Precision-Recall Curve - {self.model_type.replace("_", " ").title()}'
        )
        plt.legend(loc="lower left", fontsize=8)

        # Improve overall style
        sns.set_style("whitegrid")

        # Save the figure
        fig_name = f"{self.features}_pr_curve_{self.model_type}.png"
        plt.savefig(os.path.join(self.plot_dir, fig_name))
        plt.tight_layout()

        # Return the figure
        return fig
