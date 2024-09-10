import pandas as pd
import numpy as np

# Import your custom modules
import exploratory_data_analysis
import regression_analysis
import classification_analysis
import clustering_analysis
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    # Load the dataset
    file_path = 'nyc_airbnb.csv'
    data = load_data(file_path)

    # Perform exploratory data analysis
    exploratory_data_analysis.perform_eda(data)

    # Prepare data for regression
    X_reg = data.drop('price', axis=1)
    y_reg = data['price']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=5525)

    # Perform regression analysis
    regression_analysis.perform_regression(X_train_reg, y_train_reg, X_test_reg, y_test_reg)

    # Prepare data for classification
    X_cls = data.drop('room_type', axis=1)
    y_cls = data['room_type']
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=5525)

    # Perform classification analysis
    results_classification = classification_analysis.perform_classification(X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    print("Classification Results:")
    for classifier, result in results_classification.items():
        print(f"{classifier}:")
        for metric, value in result.items():
            if metric == 'Confusion Matrix':
                print(f"{metric}:")
                print(value)
            else:
                print(f"{metric}: {value}")
        print()


    # Perform clustering and association analysis
    n_clusters = 3
    min_support = 0.01
    cluster_labels, rules = clustering_analysis.perform_clustering_and_association_analysis(data, n_clusters, min_support)



if __name__ == "__main__":
    main()
