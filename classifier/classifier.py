import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    f1_score,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.multiclass import OneVsRestClassifier


class Classifier:
    """
    Linear Classifier for predicting experimental group based on behavioral summaries.
    """
    def __init__(
        self,
        model_type="logistic_regression",
        color_mapping=None,
        random_state=42,
        plot_dir="plots/model_plots",
        features="moseq",
    ):
        """
        Initialize Classifier.

        Args:
        - model_type (str): Type of model to use (logistic_regression, random_forest, or svm). Defaults to Logistic Regression.
        - color_mapping (dict): Dictionary mapping experimental group to specific colors. If None, it will use viridis.
        - random_state (int): Random state used to initialize the classifier.
        - plot_dir (str): Path to save plots in.
        - features (str): Prediction features being used. Defaults to MoSeq.

        Attributes:
        - model_type (str): Type of model to use (logistic_regression, random_forest, or svm). Defaults to Logistic Regression.
        - color_mapping (dict): Dictionary mapping experimental group to specific colors. If None, it will use viridis.
        - random_state (int): Random state used to initialize the classifier.
        - plot_dir (str): Path to save plots in.
        - model (OneVsRestClassifier, RandomForestClassifier, or SVC): The model that is instantiated based on the specified `model_type`.
        - scaler (sklearn.preprocessing.StandardScaler): Scaler used to standardize features so that they will have mean 0 and variance 1.
        - smote (SMOTE): Synthetic Minority Over-sampling Technique for balancing classes in training data.
        - skf (StratifiedKFold): K-fold cross-validation splitter with stratification to preserve the percentage of samples for each class.
        - features (str): Prediction features being used. Defaults to MoSeq.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=self.random_state)
        self.plot_dir = plot_dir
        self.features = features

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.skf = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=self.random_state
        )

        self.color_mapping = color_mapping if color_mapping is not None else {}

    def _get_model(self):
        """
        Instantiates a model based on the model_type passed as an argument.

        Args:
        None:

        Returns:
        Machine Learning model (Logistic Regression, Random Forest Classifier, or Support Vector Classifier) depending on self.model_type
        """
        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(
                random_state=self.random_state, max_iter=10000, penalty="l2"
            )
            return OneVsRestClassifier(base_model)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            return SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(
                "Invalid model type. Choose 'logistic_regression', 'random_forest', or 'svm'."
            )

    def _preprocess_data(self, X, y):
        """
        Helper function to scale the data with StandardScaler and handle class imbalance with SMOTE.

        Args:
        X (pandas.DataFrame): Pandas DataFrame containing the features.
        y (pandas.Series): The group column of the original fingerprint_df.

        Note: X and y must have the same length and are assumed to be indexed along the same axis).

        Returns:
        X_resampled (numpy.array): Numpy array of shape (n_samples, n_features), where n_samples might be different from X due to resampling
        y_resampled (numpy.array): Numpy array of shape (n_samples,) where n_samples matches X_resampled
        """
        X_scaled = self.scaler.fit_transform(X)
        X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
        return X_resampled, y_resampled

    def train_and_evaluate(self, X, y):
        """
        Trains, fits, and evaluates a model using cross-validation.

        This method performs the following steps:
        1. Preprocesses the data using SMOTE and StandardScaler
        2. Performs stratified k-fold cross-validation
        3. Makes predictions using the trained model
        4. Calculates various performance metrics
        5. Generates a confusion matrix plot

        Args:
        X (pandas.DataFrame): Pandas DataFrame containing the features.
        y (pandas.Series): The group column of the original fingerprint_df.

        Note: X and y must have the same length and are assumed to be indexed along the same axis.

        Returns:
        tuple: A tuple containing:
            - y_pred (numpy.ndarray): Predicted labels from cross-validation.
            - accuracy (float): Overall accuracy of the model.
            - conf_matrix (numpy.ndarray): Confusion matrix.
            - class_report (str): Classification report as a string, including precision, recall, and F1-score for each class.
            - f1 (float): Weighted F1 score.
            - cm_fig (matplotlib.figure.Figure): Confusion matrix plot as a matplotlib figure.

        Side effects:
        - Prints the model's accuracy, classification report, and F1 score to the console.
        - Saves a confusion matrix plot to the specified plot directory.

        Notes:
        - Uses the SMOTE algorithm for handling class imbalance.
        - Applies StandardScaler to normalize features.
        - Employs stratified k-fold cross-validation for more robust performance estimation.
        - The confusion matrix plot is generated using seaborn and customized for readability.
        - The method handles multi-class classification scenarios.
        """"
                                                                    
        X_resampled, y_resampled = self._preprocess_data(X, y)

        y_pred = cross_val_predict(
            self.model, X_resampled, y_resampled, cv=self.skf, n_jobs=-1
        )

        accuracy = np.mean(y_pred == y_resampled)
        conf_matrix = confusion_matrix(y_resampled, y_pred)
        class_report = classification_report(y_resampled, y_pred)
        f1 = f1_score(y_resampled, y_pred, average="weighted")

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
            )  # Adjust annotation font size and color bar size
            plt.xticks(
                rotation=45, ha="right", fontsize=7
            )  # Rotate x-axis labels and adjust font size
            plt.yticks(fontsize=7)  # Adjust y-axis label font size
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

        class_names = np.unique(y_resampled)
        cm_fig = plot_cm(conf_matrix, class_names)
        print(f"F1 Score: {f1:.4f}")

        return y_pred, accuracy, conf_matrix, class_report, f1, cm_fig

    def get_feature_importance(self, X, y):
        """
        Calculates and returns feature importance for the trained model.

        Args:
        X (pandas.DataFrame): Pandas DataFrame containing the features.
        y (pandas.Series): The group column of the original fingerprint_df.

        Returns:
        importances_df (pandas.DataFrame): A DataFrame containing features and their importance scores, sorted in descending order.

        Notes:
        - For logistic regression, feature importance is based on the absolute values of the coefficients.
        - For random forest, feature importance is based on the built-in feature_importances_ attribute.
        - For SVM, feature importance is based on the absolute values of the coefficients (only for linear kernel).
        - The method prints the top 10 feature importances and saves the full list to a CSV file.
        """
        X_resampled, y_resampled = self.preprocess_data(X, y)
        self.model.fit(X_resampled, y_resampled)

        if self.model_type == "logistic_regression":
            importances = self.model.estimators_[0].coef_[0]
        elif self.model_type == "random_forest":
            importances = self.model.feature_importances_
        elif self.model_type == "svm":
            importances = np.abs(self.model.coef_[0])

        features = X.columns
        importances_df = pd.DataFrame({"Feature": features, "Importance": importances})
        importances_df = importances_df.sort_values(by="Importance", ascending=False)

        print("Top 10 Feature Importances:")
        print(importances_df.head(10))

        save_path = os.path.join(self.plot_dir, "feature_importances.csv")
        importances_df.to_csv(save_path)

        return importances_df

    def plot_precision_recall_curve(self, X, y):
        """
        Plots the Precision-Recall curve for each class and the micro-average.

        Args:
        X (pandas.DataFrame): Pandas DataFrame containing the features.
        y (pandas.Series): The group column of the original fingerprint_df.

        Returns:
        matplotlib.figure.Figure: The generated Precision-Recall curve plot.

        Notes:
        - Uses cross-validation to compute prediction probabilities.
        - Generates a Precision-Recall curve for each class and the micro-average.
        - Uses color mapping if provided, otherwise uses a default color scheme.
        - Saves the plot as a PNG file in the specified plot directory.
        - The plot includes the Average Precision (AP) score for each class in the legend.
        """
        X_resampled, y_resampled = self.preprocess_data(X, y)

        # Use tqdm to show progress
        y_score = cross_val_predict(
            self.model,
            X_resampled,
            y_resampled,
            cv=self.skf,
            method="predict_proba",
            n_jobs=-1,
            verbose=1,
        )
        class_labels = np.unique(y_resampled)
        # Binarize the output
        y_test = label_binarize(y_resampled, classes=np.unique(y_resampled))
        n_classes = y_test.shape[1]

        # Compute PR curve and average precision for each class
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test[:, i], y_score[:, i]
            )
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

        # Compute micro-average PR curve and AP
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(
            y_test, y_score, average="micro"
        )

        # Plot the PR curve
        fig = plt.figure(figsize=(6, 5), dpi=200)
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []

        # Plot PR curve for each class
        for i, class_label in enumerate(class_labels):
            if class_label in self.color_mapping:
                color = self.color_mapping[class_label]
            else:
                color = plt.cm.jet(i / n_classes)
            display = plt.plot(
                recall[i],
                precision[i],
                color=color,
                lw=2,
                label=f"{class_label} (AP = {average_precision[i]:0.2f})",
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
