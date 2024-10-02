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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.multiclass import OneVsRestClassifier

class Classifier:
    def __init__(self, model_type='logistic_regression', color_mapping=None, random_state=42, plot_dir='plots/model_plots', features='moseq'):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=self.random_state)
        self.plot_dir = plot_dir
        self.features = features
        
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        self.color_mapping = color_mapping if color_mapping is not None else {}

    def _get_model(self):
        if self.model_type == 'logistic_regression':
            base_model = LogisticRegression(random_state=self.random_state, max_iter=10000, penalty='l2')
            return OneVsRestClassifier(base_model)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'svm':
            return SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError("Invalid model type. Choose 'logistic_regression', 'random_forest', or 'svm'.")

    def preprocess_data(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
        return X_resampled, y_resampled

    def train_and_evaluate(self, X, y):
        X_resampled, y_resampled = self.preprocess_data(X, y)
        
        y_pred = cross_val_predict(self.model, X_resampled, y_resampled, cv=self.skf, n_jobs=-1)
        
        accuracy = np.mean(y_pred == y_resampled)
        conf_matrix = confusion_matrix(y_resampled, y_pred)
        class_report = classification_report(y_resampled, y_pred)
        f1 = f1_score(y_resampled, y_pred, average='weighted')

        def plot_cm(conf_matrix, class_names):
            plt.figure(figsize=(8, 6), dpi=200)
            sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12}, cbar_kws={"shrink": 0.75})  # Adjust annotation font size and color bar size
            plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels and adjust font size
            plt.yticks(fontsize=7)  # Adjust y-axis label font size
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.title('Confusion Matrix', fontsize=14)
            plt.tight_layout()

            fig_name = f'{self.features}_confusion_matrix_{self.model_type}.png'
            plt.savefig(os.path.join(self.plot_dir, fig_name))
            plt.show()

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(class_report)
        
        class_names = np.unique(y_resampled)
        plot_cm(conf_matrix, class_names)
        print(f"F1 Score: {f1:.4f}")

        return y_pred, accuracy, conf_matrix, class_report, f1

    def get_feature_importance(self, X, y):
        X_resampled, y_resampled = self.preprocess_data(X, y)
        self.model.fit(X_resampled, y_resampled)
        
        if self.model_type == 'logistic_regression':
            importances = self.model.estimators_[0].coef_[0]
        elif self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'svm':
            importances = np.abs(self.model.coef_[0])
        
        features = X.columns
        importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importances_df = importances_df.sort_values(by='Importance', ascending=False)
        
        print("Top 10 Feature Importances:")
        print(importances_df.head(10))
        
        save_path = os.path.join(self.plot_dir, 'feature_importances.csv')
        importances_df.to_csv(save_path)
        
        return importances_df

    def plot_precision_recall_curve(self, X, y):
        X_resampled, y_resampled = self.preprocess_data(X, y)
        
        # Use tqdm to show progress
        y_score = cross_val_predict(self.model, X_resampled, y_resampled, cv=self.skf, method='predict_proba', n_jobs=-1, verbose=1)
        class_labels = np.unique(y_resampled)
        # Binarize the output
        y_test = label_binarize(y_resampled, classes=np.unique(y_resampled))
        n_classes = y_test.shape[1]

        # Compute PR curve and average precision for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

        # Compute micro-average PR curve and AP
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

        # Plot the PR curve
        plt.figure(figsize=(6, 5), dpi=200)
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        # Plot PR curve for each class
        for i, class_label in enumerate(class_labels):
            if class_label in self.color_mapping:
                color = self.color_mapping[class_label]
            else:
                color = plt.cm.jet(i / n_classes)
            display = plt.plot(recall[i], precision[i], color=color, lw=2,
                               label=f'{class_label} (AP = {average_precision[i]:0.2f})')
        

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type.replace("_", " ").title()}')
        plt.legend(loc="lower left", fontsize=8)
        
        # Improve overall style
        sns.set_style("whitegrid")
        fig_name = f'{self.features}_pr_curve_{self.model_type}.png'
        plt.savefig(os.path.join(self.plot_dir, fig_name))
        plt.tight_layout()
        
        plt.show()
