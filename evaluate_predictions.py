"""
evaluate_predictions.py

Quick evaluation script to compute F1, Accuracy, Precision, Recall
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate(y_true, y_pred):
    """Compute key metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    return metrics

def print_metrics(metrics, name="Metrics"):
    """Pretty print metrics"""
    print("\n" + "="*50)
    print(f"  {name}")
    print("="*50)
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  F1-Macro:      {metrics['f1_macro']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Load training data (with ground truth)
    train_df = pd.read_csv("data/train.csv")
    
    print("📊 Training Set Metrics (Sanity Check)")
    print("   Note: This evaluates on training data only")
    print("   (Real evaluation would be on held-out test data with labels)")
    
    # Create dummy predictions for demo
    y_train_true = train_df['target'].values
    y_train_pred = np.random.randint(0, 2, len(y_train_true))  # Random predictions
    
    train_metrics = evaluate(y_train_true, y_train_pred)
    print_metrics(train_metrics, "Random Baseline")
    
    # If you have ground truth for test set, uncomment below:
    # test_df = pd.read_csv("data/test_with_labels.csv")  # If available
    # test_preds = pd.read_csv("submissions/advanced_gnn_preds.csv")
    # y_test_true = test_df['target'].values
    # y_test_pred = test_preds['target'].values
    # test_metrics = evaluate(y_test_true, y_test_pred)
    # print_metrics(test_metrics, "Test Set Metrics")
