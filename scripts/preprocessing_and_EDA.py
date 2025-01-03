import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def test_user_behavior(data):

    # Task 1.1: User Behavior on Applications

    # Aggregate data per user
    user_agg = data.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',  # Total session duration for each user
        'Total DL (Bytes)': 'sum',  # Total download data for each user
        'Total UL (Bytes)': 'sum',  # Total upload data for each user
        'Bearer Id': 'count'  # Count of xDR sessions per user
    }).reset_index()

    # Rename columns for clarity
    user_agg.rename(columns={
        'Dur. (ms)': 'Total Session Duration (ms)',
        'Total DL (Bytes)': 'Total Download (Bytes)',
        'Total UL (Bytes)': 'Total Upload (Bytes)',
        'Bearer Id': 'Number of Sessions'
    }, inplace=True)

    # Task 1.2: Exploratory Data Analysis (EDA)

    # Handling missing values by filling with mean
    user_agg.fillna(user_agg.mean(), inplace=True)

    # Detect and handle outliers using the Interquartile Range (IQR) method
    Q1 = user_agg.quantile(0.25)
    Q3 = user_agg.quantile(0.75)
    IQR = Q3 - Q1
    user_agg = user_agg[~((user_agg < (Q1 - 1.5 * IQR)) | (user_agg > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Descriptive statistics for the aggregated user data
    desc_stats = user_agg.describe()

    # Segment users into decile classes based on total session duration
    user_agg['Duration Decile'] = pd.qcut(user_agg['Total Session Duration (ms)'], 5, labels=False) + 1
    data_per_decile = user_agg.groupby('Duration Decile').agg({
        'Total Download (Bytes)': 'sum',
        'Total Upload (Bytes)': 'sum'
    }).reset_index()

    # Univariate Analysis
    # Non-Graphical: Calculate dispersion parameters
    dispersion_params = user_agg[['Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']].agg(['mean', 'median', 'std'])

    # Graphical Analysis
    sns.histplot(user_agg['Total Session Duration (ms)'], bins=30, kde=True)
    plt.title('Total Session Duration Distribution')
    plt.show()

    # Bivariate Analysis: Scatter plot of Download vs Upload Data
    sns.scatterplot(x='Total Download (Bytes)', y='Total Upload (Bytes)', data=user_agg)
    plt.title('Download vs Upload Data')
    plt.show()

    # Correlation Analysis: Heatmap of correlation matrix
    correlation_matrix = user_agg.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Dimensionality Reduction using PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(user_agg[['Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Task 2: User Engagement Analysis

    # Calculate engagement metrics
    engagement_metrics = data.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Sessions frequency per user
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()

    # Rename columns for clarity
    engagement_metrics.rename(columns={
        'Bearer Id': 'Sessions Frequency',
        'Dur. (ms)': 'Total Session Duration (ms)',
        'Total DL (Bytes)': 'Total Download (Bytes)',
        'Total UL (Bytes)': 'Total Upload (Bytes)'
    }, inplace=True)

    # Normalize engagement metrics for clustering
    engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']] = \
        (engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']] - 
        engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']].min()) / \
        (engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']].max() - 
        engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']].min())

    # K-Means Clustering to identify user segments
    kmeans = KMeans(n_clusters=3)
    engagement_metrics['Cluster'] = kmeans.fit_predict(engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']])

    # Summarize cluster metrics
    cluster_summary = engagement_metrics.groupby('Cluster').agg({
        'Sessions Frequency': ['min', 'max', 'mean', 'sum'],
        'Total Session Duration (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Download (Bytes)': ['min', 'max', 'mean', 'sum'],
        'Total Upload (Bytes)': ['min', 'max', 'mean', 'sum'],
    }).reset_index()

    # Identify top engaged users per application
    top_users_per_app = data.groupby('Handset Manufacturer').agg({
        'Total DL (Bytes)': 'sum',  # Total download data per manufacturer
    }).reset_index().nlargest(10, 'Total DL (Bytes)')

    # Plotting top applications
    top_apps = top_users_per_app.nlargest(3, 'Total DL (Bytes)')
    sns.barplot(x='Handset Manufacturer', y='Total DL (Bytes)', data=top_apps)
    plt.title('Top 3 Most Engaged Handset Manufacturers')
    plt.show()

    # K-Means Optimization using Elbow Method
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(engagement_metrics[['Sessions Frequency', 'Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']])
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Method results
    plt.plot(range(1, 10), inertia)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()