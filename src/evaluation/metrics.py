import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

class ModelEvaluator:
    """Class for evaluating fake news detection models"""
    
    def __init__(self, model, X_test, y_test):
        """Initialize the evaluator
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_prob = None
        
    def predict(self):
        """Make predictions using the model
        
        Returns:
            Predicted labels
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
    
    def predict_proba(self):
        """Predict class probabilities using the model
        
        Returns:
            Class probabilities
        """
        try:
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        except (AttributeError, IndexError):
            # If predict_proba is not available or returns unexpected format
            self.y_prob = self.model.predict(self.X_test)
        return self.y_prob
    
    def confusion_matrix(self, normalize=False):
        """Compute the confusion matrix
        
        Args:
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Confusion matrix
        """
        if self.y_pred is None:
            self.predict()
            
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        return cm
    
    def plot_confusion_matrix(self, normalize=False, figsize=(10, 8)):
        """Plot the confusion matrix
        
        Args:
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = self.confusion_matrix(normalize=normalize)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd', 
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        return plt.gcf()
    
    def classification_report_df(self):
        """Generate a classification report as a DataFrame
        
        Returns:
            DataFrame with classification metrics
        """
        if self.y_pred is None:
            self.predict()
            
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()
    
    def roc_curve(self):
        """Compute the ROC curve
        
        Returns:
            fpr, tpr, thresholds, roc_auc
        """
        if self.y_prob is None:
            self.predict_proba()
            
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        return fpr, tpr, thresholds, roc_auc
    
    def plot_roc_curve(self, figsize=(10, 8)):
        """Plot the ROC curve
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _, roc_auc = self.roc_curve()
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def precision_recall_curve(self):
        """Compute the precision-recall curve
        
        Returns:
            precision, recall, thresholds, average_precision
        """
        if self.y_prob is None:
            self.predict_proba()
            
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        average_precision = average_precision_score(self.y_test, self.y_prob)
        
        return precision, recall, thresholds, average_precision
    
    def plot_precision_recall_curve(self, figsize=(10, 8)):
        """Plot the precision-recall curve
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _, average_precision = self.precision_recall_curve()
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def evaluate(self):
        """Perform a comprehensive evaluation of the model
        
        Returns:
            Dictionary with evaluation results
        """
        if self.y_pred is None:
            self.predict()
        if self.y_prob is None:
            self.predict_proba()
            
        # Classification report
        report_df = self.classification_report_df()
        
        # ROC curve
        fpr, tpr, _, roc_auc = self.roc_curve()
        
        # Precision-recall curve
        precision, recall, _, average_precision = self.precision_recall_curve()
        
        # Confusion matrix
        cm = self.confusion_matrix()
        
        return {
            'classification_report': report_df,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr
            },
            'precision_recall_curve': {
                'precision': precision,
                'recall': recall
            }
        }
    
    def plot_evaluation_results(self, figsize=(20, 15)):
        """Plot all evaluation results
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.y_pred is None:
            self.predict()
        if self.y_prob is None:
            self.predict_proba()
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Confusion matrix
        cm = self.confusion_matrix()
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=axes[0, 0]
        )
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_title('Confusion Matrix')
        
        # ROC curve
        fpr, tpr, _, roc_auc = self.roc_curve()
        axes[0, 1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-recall curve
        precision, recall, _, average_precision = self.precision_recall_curve()
        axes[1, 0].plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc='lower left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Classification report as a table
        report_df = self.classification_report_df()
        axes[1, 1].axis('off')
        axes[1, 1].table(
            cellText=report_df.values.round(2),
            rowLabels=report_df.index,
            colLabels=report_df.columns,
            cellLoc='center',
            loc='center'
        )
        axes[1, 1].set_title('Classification Report')
        
        plt.tight_layout()
        return fig