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

Author
Natnahom Asfaw
23/12/2024
