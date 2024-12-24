import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Database connection
# def connect_to_db():
#     db_host = os.getenv('DB_HOST')
#     db_port = os.getenv('DB_PORT')
#     db_name = os.getenv('DB_NAME')
#     db_user = os.getenv('DB_USER')
#     db_password = os.getenv('DB_PASSWORD')
    
#     connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
#     engine = create_engine(connection_string)
#     return engine

# load_data_using_sqlalchemy()

# Task 3.1: Aggregate user experience metrics
def experience_analytics(data):
    """
    Aggregates user experience metrics from the provided data.
    
    Args:
        data (pd.DataFrame): Raw data containing user metrics.

    Returns:
        pd.DataFrame: Aggregated user metrics per MSISDN/Number.
    """
    # Check the actual column names for debugging
    print(data.columns)

    # Replace missing values with mean/mode for relevant columns
    data['TCP DL Retrans. Vol (Bytes)'] = data['TCP DL Retrans. Vol (Bytes)'].fillna(data['TCP DL Retrans. Vol (Bytes)'].mean())
    data['TCP UL Retrans. Vol (Bytes)'] = data['TCP UL Retrans. Vol (Bytes)'].fillna(data['TCP UL Retrans. Vol (Bytes)'].mean())
    data['Avg RTT DL (ms)'] = data['Avg RTT DL (ms)'].fillna(data['Avg RTT DL (ms)'].mean())
    data['Avg RTT UL (ms)'] = data['Avg RTT UL (ms)'].fillna(data['Avg RTT UL (ms)'].mean())
    data['Avg Bearer TP DL (kbps)'] = data['Avg Bearer TP DL (kbps)'].fillna(data['Avg Bearer TP DL (kbps)'].mean())
    data['Avg Bearer TP UL (kbps)'] = data['Avg Bearer TP UL (kbps)'].fillna(data['Avg Bearer TP UL (kbps)'].mean())
    data['Total DL (Bytes)'] = data['Total DL (Bytes)'].fillna(data['Total DL (Bytes)'].mean())
    data['Handset Type'] = data['Handset Type'].fillna(data['Handset Type'].mode()[0])

    # Aggregate metrics by user
    user_metrics = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Total DL (Bytes)': 'mean',
        'Handset Type': 'first'  # Assuming the first entry is representative
    }).reset_index()

    return user_metrics

# Task 3.2: Compute top, bottom, and most frequent values
def compute_top_bottom_frequent(user_metrics):
    """
    Computes top, bottom, and most frequent values for specified metrics.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        dict: Top, bottom, and frequent values for various metrics.
    """
    top_tcp = user_metrics['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
    bottom_tcp = user_metrics['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
    freq_tcp = user_metrics['TCP DL Retrans. Vol (Bytes)'].value_counts().nlargest(10)

    top_rtt = user_metrics['Avg RTT DL (ms)'].nlargest(10)
    bottom_rtt = user_metrics['Avg RTT DL (ms)'].nsmallest(10)
    freq_rtt = user_metrics['Avg RTT DL (ms)'].value_counts().nlargest(10)

    top_throughput = user_metrics['Avg Bearer TP DL (kbps)'].nlargest(10)
    bottom_throughput = user_metrics['Avg Bearer TP DL (kbps)'].nsmallest(10)
    freq_throughput = user_metrics['Avg Bearer TP DL (kbps)'].value_counts().nlargest(10)

    return {
        'TCP DL Retrans. Vol (Bytes)': (top_tcp, bottom_tcp, freq_tcp),
        'Avg RTT DL (ms)': (top_rtt, bottom_rtt, freq_rtt),
        'Avg Bearer TP DL (kbps)': (top_throughput, bottom_throughput, freq_throughput)
    }

# Task 3.3: Distribution and average metrics by handset type
def distribution_by_handset(user_metrics):
    """
    Analyzes and visualizes metrics distribution by handset type.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        tuple: Mean throughput and TCP retransmissions by handset type.
    """
    # Calculate average metrics by handset type
    throughput_by_handset = user_metrics.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
    tcp_by_handset = user_metrics.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()

    # Boxplots for visual analysis
    sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=user_metrics)
    plt.title('Throughput Distribution by Handset Type')
    plt.xticks(rotation=45)
    plt.show()

    sns.boxplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=user_metrics)
    plt.title('TCP Retransmission by Handset Type')
    plt.xticks(rotation=45)
    plt.show()

    return throughput_by_handset, tcp_by_handset

# Task 3.4: K-Means clustering
def kmeans_clustering(user_metrics):
    """
    Performs K-Means clustering on user metrics.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        pd.DataFrame: User metrics with cluster labels.
    """
    features = user_metrics[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_metrics['Cluster'] = kmeans.fit_predict(scaled_features)

    return user_metrics

# Task 4.1: Assign engagement and experience scores
def assign_scores(user_metrics, engagement_metrics):
    """
    Assigns engagement and experience scores based on user metrics.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.
        engagement_metrics (pd.DataFrame): DataFrame containing engagement metrics.

    Returns:
        pd.DataFrame: User metrics with assigned scores.
    """
    # Print column names for debugging
    print("User Metrics Columns:", user_metrics.columns)
    print("Engagement Metrics Columns:", engagement_metrics.columns)

    # Clean up column names to avoid issues with spaces
    user_metrics.columns = user_metrics.columns.str.strip()

    # Check if engagement_metrics is not empty before proceeding
    if engagement_metrics.empty:
        print("Warning: engagement_metrics is empty.")
        return None  # Early return if engagement_metrics is empty

    engagement_metrics.columns = engagement_metrics.columns.str.strip()

    # Check if merge keys exist
    if 'MSISDN/Number' not in user_metrics.columns:
        print("Error: 'MSISDN/Number' not found in user_metrics.")
        return None  # Handle this case as appropriate

    if 'MSISDN/Number' not in engagement_metrics.columns:
        print("Error: 'MSISDN/Number' not found in engagement_metrics.")
        return None  # Handle this case as appropriate

    # Perform the merge on MSISDN/Number
    user_metrics = user_metrics.merge(engagement_metrics, on='MSISDN/Number', how='left')

    # Check if the merge resulted in any data
    if user_metrics.empty:
        print("Warning: No data found after merging.")
        return None

    # Calculate Euclidean distances for engagement and experience scores
    engagement_scores = pairwise_distances(user_metrics[['Total DL (Bytes)']], 
                                            user_metrics[['Total DL (Bytes)']].min(axis=0).values.reshape(1, -1))
    experience_scores = pairwise_distances(user_metrics[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']], 
                                            user_metrics[user_metrics['Cluster'] == user_metrics['Cluster'].min()][['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']].mean().values.reshape(1, -1))

    user_metrics['Engagement Score'] = engagement_scores
    user_metrics['Experience Score'] = experience_scores

    return user_metrics

# Task 4.2: Satisfaction score calculation
def calculate_satisfaction(user_metrics):
    """
    Calculates the satisfaction score based on engagement and experience scores.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        pd.DataFrame: Top satisfied users based on satisfaction score.
    """
    user_metrics['Satisfaction Score'] = (user_metrics['Engagement Score'] + user_metrics['Experience Score']) / 2
    top_satisfied = user_metrics.nlargest(10, 'Satisfaction Score')

    return top_satisfied

# Task 4.3: Regression model for satisfaction score
def regression_model(user_metrics):
    """
    Builds a linear regression model to predict satisfaction score.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        LinearRegression: Trained regression model.
    """
    X = user_metrics[['Engagement Score', 'Experience Score']]
    y = user_metrics['Satisfaction Score']
    
    model = LinearRegression()
    model.fit(X, y)

    return model

# Task 4.4: K-Means on engagement and experience scores
def kmeans_on_scores(user_metrics):
    """
    Performs K-Means clustering on engagement and experience scores.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        pd.DataFrame: User metrics with score clusters.
    """
    scores = user_metrics[['Engagement Score', 'Experience Score']]
    kmeans = KMeans(n_clusters=2)
    user_metrics['Score Cluster'] = kmeans.fit_predict(scores)

    return user_metrics

# Task 4.5: Average satisfaction and experience score per cluster
def average_scores_per_cluster(user_metrics):
    """
    Calculates the average satisfaction and experience scores per cluster.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.

    Returns:
        pd.DataFrame: Average scores per cluster.
    """
    avg_scores = user_metrics.groupby('Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()

    return avg_scores

# Task 4.6: Export final table to MySQL
def export_to_mysql(user_metrics):
    """
    Exports the final user metrics DataFrame to a MySQL database.
    
    Args:
        user_metrics (pd.DataFrame): DataFrame containing user metrics.
    """
    engine = connect_to_db()
    user_metrics.to_sql('satisfaction_analysis', con=engine, if_exists='replace', index=False)