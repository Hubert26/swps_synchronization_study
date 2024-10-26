# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:03:38 2024

@author: Hubert Szewczyk
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, precision_recall_fscore_support)
from sklearn.metrics import confusion_matrix, roc_auc_score

#import eli5
#from eli5.sklearn import PermutationImportance
from sklearn.model_selection import cross_val_score

#%%
def bootstrap_auc(y_true, y_pred_prob, n_bootstraps=1000, alpha=0.95):
    """
    Calculates the bootstrap confidence intervals for the Area Under the ROC Curve (AUC).
    
    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_prob: Array-like, predicted probabilities for the positive class.
    - n_bootstraps: Integer, the number of bootstrap samples to generate (default is 1000).
    - alpha: Float, the confidence level for the interval (default is 0.95 for 95% confidence interval).

    Returns:
    - lower_bound: Float, the lower bound of the bootstrap confidence interval for AUC.
    - upper_bound: Float, the upper bound of the bootstrap confidence interval for AUC.

    Raises:
    - ValueError: If no valid bootstrap samples could be drawn (e.g., only one class in every sample).
    """
    # Ensure y_true and y_pred_prob are 1D arrays
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    # Initialize random state for reproducibility
    rng = np.random.RandomState(seed=42)
    bootstrapped_aucs = []

    for i in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
        if len(np.unique(y_true[indices])) < 2:
            # Skip samples where one of the classes is missing
            continue

        # Compute AUC for the current bootstrap sample
        score = roc_auc_score(y_true[indices], y_pred_prob[indices])
        bootstrapped_aucs.append(score)

    # Ensure at least one valid bootstrap sample was created
    if len(bootstrapped_aucs) == 0:
        raise ValueError("All bootstrap samples were invalid (contained only one class).")

    # Sort the bootstrapped AUC scores
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()

    # Compute confidence intervals based on the sorted scores
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)

    return lower_bound, upper_bound

#%%
def risk_group_metrics(y_test, y_pred_prob, threshold):
    """
    Calculates various risk metrics for a given threshold on predicted probabilities.

    Parameters:
    - y_test: Array-like, true labels for the test set.
    - y_pred_prob: Array-like, predicted probabilities for the positive class.
    - threshold: Float, the risk threshold to apply for classifying high-risk observations.

    Returns:
    - A dictionary of calculated metrics including confusion matrix values, AUROC, false positive rate (FPR),
      false negative rate (FNR), specificity, and risk ratios for the high-risk group.
    """
    # Identifying high-risk instances based on the threshold
    high_risk = y_pred_prob >= threshold
    y_pred_high_risk = np.zeros_like(y_pred_prob)
    y_pred_high_risk[high_risk] = 1  # Assigning class 1 to high-risk instances

    # Filtering y_test and y_pred_prob for high-risk observations
    y_test_high_risk = y_test[high_risk]
    y_pred_prob_high_risk = y_pred_prob[high_risk]

    # Check if we have enough samples to calculate the confusion matrix
    if len(np.unique(y_test_high_risk)) < 2:
        # Return NaN if only one class exists in the high-risk data
        tn = fp = fn = tp = np.nan
        risk_ratio = low_risk_positive_rate = high_risk_positive_rate = specificity = roc_score = lower = upper = fpr = fnr = np.nan
    else:
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(y_test, y_pred_high_risk)
        tn, fp, fn, tp = cm.ravel()  # Extract true negatives, false positives, etc.

        # Calculate false positive rate, false negative rate, and specificity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate positive rates for high and low risk
        high_risk_positive_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        low_risk_positive_rate = fn / (fn + tn) if (fn + tn) > 0 else 0

        # Calculate risk ratio
        risk_ratio = high_risk_positive_rate / low_risk_positive_rate if low_risk_positive_rate > 0 else 0

        # Calculate AUROC and bootstrap confidence intervals
        roc_score = roc_auc_score(y_test_high_risk, y_pred_prob_high_risk)
        lower, upper = bootstrap_auc(y_test_high_risk, y_pred_prob_high_risk, n_bootstraps=1000, alpha=0.95)

    return {
        'class_0_pred': (y_pred_high_risk == 0).sum(),  # Number of predicted class 0
        'class_1_pred': (y_pred_high_risk == 1).sum(),  # Number of predicted class 1
        'AUROC': roc_score,  # Area under ROC curve
        'AUROClow': lower,  # Lower bound of AUROC CI
        'AUROCup': upper,  # Upper bound of AUROC CI
        'tn': tn,  # True negatives
        'fp': fp,  # False positives
        'fn': fn,  # False negatives
        'tp': tp,  # True positives
        'fpr': fpr,  # False positive rate
        'fnr': fnr,  # False negative rate
        'specificity': specificity,  # Specificity
        'high_risk_positive_rate': high_risk_positive_rate,  # Positive rate in high-risk group
        'low_risk_positive_rate': low_risk_positive_rate,  # Positive rate in low-risk group
        'risk_ratio': risk_ratio  # Risk ratio
    }

#%%
def model_validation(model, X_test, y_test, risk_thresholds=[0.5, 0.9, 0.95, 0.99]):
    """
    Validates the performance of a trained model on test data.

    Parameters:
    -----------
    model : object
        Trained machine learning model with `predict` and `predict_proba` methods.
    X_test : array-like
        Feature data for testing.
    y_test : array-like
        True labels for the test data.
    risk_thresholds : list of floats, optional
        List of thresholds to assess risk-group metrics, default is [0.5, 0.9, 0.95, 0.99].

    Returns:
    --------
    model_results : pandas.DataFrame
        DataFrame containing model performance metrics such as accuracy, precision, recall, and F1-score.
    """
    
    # Check if y_test and X_test are not empty
    if y_test is None or X_test is None or len(y_test) == 0 or len(X_test) == 0:
        raise ValueError("y_test and X_test must not be empty.")
    
    # Check if the sizes of X_test and y_test match
    if len(X_test) != len(y_test):
        raise ValueError(f"Inconsistent number of samples: X_test has {len(X_test)} samples, y_test has {len(y_test)} samples.")

    # Predict labels and probabilities
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification (probability for class 1)

    # Compute basic performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Create a DataFrame for overall model results
    model_results = pd.DataFrame({
        'recall': [recall],
        'accuracy': [accuracy],
        'precision': [precision],
        'f1': [f1],
    })

    # Calculate precision, recall, f1-score, and support for each class
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

    # Add individual class results to the DataFrame
    for idx, class_ in enumerate(np.unique(y_test)):
        model_results[f'precision_{class_}'] = class_precision[idx]
        model_results[f'recall_{class_}'] = class_recall[idx]
        model_results[f'f1_{class_}'] = class_f1[idx]
        model_results[f'support_{class_}'] = support[idx]

    # Calculate risk group metrics for each threshold
    for threshold in risk_thresholds:
        risk_metrics = risk_group_metrics(y_test, y_pred_prob, threshold)
        for key, value in risk_metrics.items():
            model_results[f'{key}_{threshold}'] = value

    return model_results

#%%
def compute_feature_importances(model, X, sort=True):
    """
    Compute feature importances for a given model and feature set X.
    
    Parameters:
    - model: Trained model object (e.g., DecisionTreeClassifier, RandomForestClassifier) that supports feature importance extraction.
    - X: pd.DataFrame, Feature matrix (must match the model's training features).
    - sort: bool, Whether to sort the features by importance in descending order (default: True).
    
    Returns:
    - importances_df: pd.DataFrame, DataFrame containing feature names and their respective importance scores.
                      Sorted by importance if sort=True.
    """
    
    # Check if the model has a feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(f"The model {type(model).__name__} does not have feature importances.")
    
    # Extract feature importances
    importances = model.feature_importances_
    
    # Sort feature importances and corresponding feature names if required
    if sort:
        sorted_indices = importances.argsort()[::-1]
        data = {'feature': [X.columns[index] for index in sorted_indices],
                'mean_decrease_impurity': [importances[index] for index in sorted_indices]}
    else:
        data = {'feature': X.columns, 'mean_decrease_impurity': importances}
    
    # Create a DataFrame of the importances
    importances_df = pd.DataFrame(data)
    
    return importances_df

#%%
# =============================================================================
# def calculate_permutation_importance(model, X, y, random_state=42):
#     """
#     Calculate the Permutation Importance for a given model.
# 
#     Parameters:
#     - model: The trained model (e.g., DecisionTreeClassifier or RandomForestClassifier).
#     - X: Features DataFrame used for prediction.
#     - y: Target values corresponding to the features.
#     - random_state: Seed for reproducibility (default is 42).
# 
#     Returns:
#     - perm_df: DataFrame containing the permutation importance of features.
#     """
#     # Compute Permutation Importance
#     perm = PermutationImportance(model, random_state=random_state).fit(X, y)
#     
#     # Extract the permutation importance results into a DataFrame
#     perm_df = eli5.explain_weights_df(perm, feature_names=list(X.columns))
#     
#     # Rename columns for clarity
#     perm_df = perm_df.rename(columns={'weight': 'Permutation_Importance_weight', 
#                                       'std': 'Permutation_Importance_std'})
#     
#     return perm_df
# 
# =============================================================================
#%%
def mean_decrease_accuracy(model, X, y):
    """
    Computes Mean Decrease Accuracy (MDA) for feature importance using permutation.

    Parameters:
    - model: The machine learning model to evaluate (must have a `fit` method).
    - X: DataFrame with feature data.
    - y: Series or array with target data.

    Returns:
    - Dictionary with features as keys and their Mean Decrease Accuracy as values.
    """

    # Calculate the baseline accuracy of the model
    baseline_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

    feature_importances = {}
    
    # Iterate through each feature to compute its Mean Decrease Accuracy
    for feature in X.columns:
        # Create a permuted copy of the dataset for the current feature
        X_permuted = X.copy()
        X_permuted[feature] = np.random.permutation(X[feature].values)
        
        # Calculate the accuracy of the model on the permuted dataset
        permuted_accuracy = cross_val_score(model, X_permuted, y, cv=5, scoring='accuracy').mean()
        
        # Compute the Mean Decrease Accuracy for the current feature
        feature_importances[feature] = baseline_accuracy - permuted_accuracy

    return pd.DataFrame(list(feature_importances.items()), columns=['feature', 'mean_decrease_accuracy'])
