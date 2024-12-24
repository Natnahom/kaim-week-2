PostgreSQL Data Loader and User Behavior Analysis
This project provides functionality to load data from a PostgreSQL database and perform user behavior analysis using Python. It includes two main components: data loading and user behavior analysis.

Requirements
Python 3.9.13
Packages:
psycopg2
pandas
sqlalchemy
python-dotenv
numpy
seaborn
matplotlib
scikit-learn
You can install the required packages using pip:

pip install psycopg2 pandas sqlalchemy python-dotenv numpy seaborn matplotlib scikit-learn

Setup
Environment Variables: Create a .env file in the root directory of your project with the following contents:
plaintext
Copy
DB_HOST=your_database_host
DB_PORT=your_database_port
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password

Replace the placeholder values with your actual database credentials.
Usage: You can use the provided functions to load data from your PostgreSQL database and analyze user behavior.
Data Loading Functions
load_data_from_postgres(query)
Connects to the PostgreSQL database and loads data based on the provided SQL query.
Returns a DataFrame containing the query results or None if an error occurs.
load_data_using_sqlalchemy(query)
Connects to the PostgreSQL database using SQLAlchemy and loads data based on the provided SQL query.
Returns a DataFrame containing the query results or None if an error occurs.
User Behavior Analysis Function
test_user_behavior(data)
This function performs a comprehensive analysis of user behavior based on the provided dataset.

User Aggregation: It aggregates user data to compute total session duration, total download, and upload data, and counts the number of sessions for each user.
Exploratory Data Analysis (EDA):
Handles missing values by filling them with the mean.
Detects and removes outliers using the Interquartile Range (IQR) method.
Provides descriptive statistics of the aggregated user data.
User Segmentation: Segments users into deciles based on total session duration to analyze user behavior across different engagement levels.
Visualizations:
- Univariate Analysis: Displays the distribution of total session duration.
- Bivariate Analysis: Creates scatter plots to visualize relationships between download and upload data.
- Correlation Analysis: Generates a heatmap to show correlation between different metrics.
- Dimensionality Reduction: Applies PCA (Principal Component Analysis) to reduce the dimensionality of user behavior metrics, helping to visualize and understand user engagement patterns.
- User Engagement Metrics: Calculates engagement metrics per user, including session frequency and total data usage.
- Clustering: Implements K-Means clustering to categorize users based on their engagement metrics and visualizes the results.
- Optimization: Uses the Elbow Method to find the optimal number of clusters for K-Means.
- Top Engaged Users: Identifies and visualizes the top engaged users per application based on download data.

Example Usage
python

query = "SELECT * FROM your_table_name"
data = load_data_from_postgres(query)

if data is not None:
    test_user_behavior(data)

There are queries int the sql_queries.py file that I used in the notebook jupyter.

PostgreSQL Data Loader and User Behavior Analysis
This project provides functionality to load data from a PostgreSQL database and perform user behavior, experience, and satisfaction analysis using Python. It includes components for data loading, user behavior analysis, experience analytics, and satisfaction analysis.

Requirements
Python 3.x
Packages:
psycopg2
pandas
sqlalchemy
python-dotenv
numpy
seaborn
matplotlib
scikit-learn
statsmodels (for regression modeling)
You can install the required packages using pip:

Copy
pip install psycopg2 pandas sqlalchemy python-dotenv numpy seaborn matplotlib scikit-learn statsmodels
Setup
Environment Variables: Create a .env file in the root directory of your project with the following contents:

Usage: You can use the provided functions to load data from your PostgreSQL database and analyze user behavior, experience, and satisfaction.
Data Loading Functions
load_data_from_postgres(query)
Connects to the PostgreSQL database and loads data based on the provided SQL query.
Returns a DataFrame containing the query results or None if an error occurs.
load_data_using_sqlalchemy(query)
Connects to the PostgreSQL database using SQLAlchemy and loads data based on the provided SQL query.
Returns a DataFrame containing the query results or None if an error occurs.
User Behavior Analysis Function
test_user_behavior(data)
This function performs a comprehensive analysis of user behavior based on the provided dataset. It includes user aggregation, exploratory data analysis (EDA), user segmentation, visualizations, dimensionality reduction, engagement metrics calculation, K-Means clustering, and identification of top engaged users.

Experience Analytics
experience_analytics(data)
This function focuses on analyzing user experience based on network parameters and device characteristics.

Task 3.1: Aggregates average TCP retransmission, RTT, handset type, and throughput per customer. It handles missing values and outliers by replacing them with the mean or mode.
Task 3.2: Computes and lists the top, bottom, and most frequent TCP, RTT, and throughput values in the dataset.
Task 3.3: Analyzes the distribution of average throughput and average TCP retransmission by handset type, providing interpretations of the findings.
Task 3.4: Performs K-Means clustering (k=3) to segment users into experience groups and provides descriptions for each cluster based on the analysis.
Satisfaction Analysis
satisfaction_analysis(data, engagement_data)
This function analyzes customer satisfaction based on engagement and experience metrics.

Task 4.1: Assigns engagement and experience scores to each user based on the Euclidean distance from the respective clusters.
Task 4.2: Computes the average of both scores as the satisfaction score and reports the top 10 satisfied customers.
Task 4.3: Builds a regression model to predict customer satisfaction scores.
Task 4.4: Runs K-Means clustering (k=2) on engagement and experience scores.
Task 4.5: Aggregates average satisfaction and experience scores per cluster.
Task 4.6: Exports the final table containing all user IDs along with engagement, experience, and satisfaction scores to a local MySQL database and provides a screenshot of the output from a select query on the exported table.

Author
Natnahom Asfaw
23/12/2024
