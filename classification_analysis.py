import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5525)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

def perform_classification(X_train, y_train, X_test, y_test):
    classifiers = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC(probability=True)),
        ('Naïve Bayes', GaussianNB()),
        ('Random Forest', RandomForestClassifier()),
        ('Neural Network', MLPClassifier(max_iter=1000))
    ]

    results = {}

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

        cm = confusion_matrix(y_test, y_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
        specificity = cm.diagonal() / cm.sum(axis=1)
        specificity = np.mean(specificity)

        y_prob = clf.predict_proba(X_test)
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc="lower right")
        plt.show()

        results[name] = {
            'Confusion Matrix': cm,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F-score': f_score,
            'Accuracy': accuracy,
            'Report': report
        }

    return results
