import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules


def k_means_clustering(data, n_clusters):
    # Preprocess the data for K-means clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_

    return cluster_labels


def binarize_data(data, threshold_dict, categorical_columns):
    binarized_data = data.copy()
    for column in data.columns:
        if column in categorical_columns:
            dummies = pd.get_dummies(data[column], prefix=column)
            binarized_data = pd.concat([binarized_data, dummies], axis=1)
            binarized_data = binarized_data.drop(columns=[column])
        elif column in threshold_dict:
            binarized_data[column] = (data[column] >= threshold_dict[column]).astype(int)
    return binarized_data.astype(bool)


def perform_apriori_algorithm(data, min_support):
    # Perform Apriori algorithm
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    return rules


def perform_clustering_and_association_analysis(data, n_clusters=3, min_support=0.01):
    # Perform K-means clustering
    cluster_labels = k_means_clustering(data, n_clusters)

    # Add the cluster labels to the data
    data['Cluster'] = cluster_labels

    # Binarize the dataset using custom thresholds
    threshold_dict = {
        'price': 100,
        'minimum_nights': 3,
        'number_of_reviews': 10
    }
    categorical_columns = ['neighbourhood_group', 'room_type']
    binarized_data = binarize_data(data, threshold_dict, categorical_columns)

    # Remove the "Cluster" column from the binarized data
    binarized_data = binarized_data.drop(columns=['Cluster'])

    # Perform Apriori algorithm
    rules = perform_apriori_algorithm(binarized_data, min_support)

    # Print the results
    print("KMeans:")
    print(cluster_labels)
    print("Apriori:")
    print(rules)

    return cluster_labels, rules
