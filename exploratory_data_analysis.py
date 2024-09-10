import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor

from sklearn.impute import SimpleImputer


def data_cleaning(data):

    data.drop(['id', 'name', 'host_id', 'host_name', 'last_review', 'latitude', 'longitude', 'number_of_reviews',
               'calculated_host_listings_count'], axis=1, inplace=True)

    # Fill missing or NaN values with suitable values or drop them
    imputer = SimpleImputer(strategy='median')
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # Convert non-numeric features to numeric using LabelEncoder
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    return data


def dimensionality_reduction(data):
    # Perform Random Forest Analysis
    print("Random Forest Analysis:")
    target = 'price'  # change 'room_type' to 'price'
    attributes = data.drop(target, axis=1).columns
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(data[attributes], data[target])
    feature_importances = pd.Series(rf.feature_importances_, index=attributes)
    top_features = feature_importances.sort_values(ascending=False).head(10)
    print(top_features)


    # Perform Principal Component Analysis
    print("Principal Component Analysis:")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[attributes])
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Perform Singular Value Decomposition Analysis
    print("Singular Value Decomposition Analysis:")
    U, s, VT = np.linalg.svd(data[attributes], full_matrices=False)
    print("Singular values:", s)

    return top_features.index, pca_result, s


def discretization_binarization(data):
    # One-hot encoding for categorical variables
    data = pd.get_dummies(data, columns=['room_type'], prefix=['room_type'])
    print("One-Hot Encoding:")
    print(data.head())


def variable_transformation(data):
    # Normalization/Standardization
    attributes = data.drop('room_type', axis=1).columns
    scaler = StandardScaler()
    data.loc[:, attributes] = scaler.fit_transform(data[attributes])
    print("Standardized Data:")
    print(data[attributes].head())


def anomaly_detection_outlier_analysis(data):
    # Anomaly detection using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outliers = data[lof.fit_predict(data) == -1]
    print("Outliers detected:")
    print(outliers)


def covariance_heatmap(data):
    # Sample covariance matrix heatmap
    cov_matrix = data.sample(frac=0.1).cov()
    plt.figure(figsize=(10, 8))  # adjust the figure size as desired
    sns.set(font_scale=1.0)  # adjust the font size as desired
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Sample Covariance Matrix Heatmap")
    plt.show()

def pearson_correlation_heatmap(data):
    # Sample Pearson correlation coefficients matrix heatmap
    corr_matrix = data.sample(frac=0.1).corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Sample Pearson Correlation Coefficients Matrix Heatmap")
    plt.show()


def perform_eda(data):
    # Set figure size and font size
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set(font_scale=1.2)

    # Data cleaning
    data = data_cleaning(data)
    print("First five rows:")
    print(data.head())
    print("Missing values:")
    print(data.isnull().sum())

    # Dimensionality reduction
    top_features, pca_result, svd_result = dimensionality_reduction(data)
    data = data[top_features.tolist() + ['price']]  # Keep only the top 10 features and the target variable

    # Discretization and binarization
    discretization_binarization(data)

    # Variable transformation
    variable_transformation(data)

    # Anomaly detection and outlier analysis
    anomaly_detection_outlier_analysis(data)

    # Sample covariance matrix heatmap
    covariance_heatmap(data)

    # Sample Pearson correlation coefficients matrix heatmap
    pearson_correlation_heatmap(data)