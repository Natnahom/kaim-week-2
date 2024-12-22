import os
import pandas as pd
import streamlit as st
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load environment variables from .env file
load_dotenv()

# Access the variables
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

st.set_page_config(page_title="Simple Dashboard", layout="wide")

with st.sidebar:
    selected_option = st.radio("Selected Option:", ["Upload SQL File", "Sample Data Preview"])

st.title("Simple Streamlit Dashboard")

if selected_option == "Upload SQL File":
    st.header("Option 1: Upload SQL File and Plot Data")

    uploaded_file = st.file_uploader("Choose an SQL file to upload", type="sql")

    if uploaded_file is not None:
        sql_query = uploaded_file.read().decode("utf-8")
        st.subheader("SQL Query Preview")
        st.code(sql_query)

        if st.button("Execute Query"):
            try:
                # Establishing the connection
                connection = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_password
                )

                # Using pandas to load the data
                data = pd.read_sql_query(sql_query, connection)
                connection.close()

                st.subheader("Fetched Data Preview")
                st.dataframe(data.head())

                if not data.empty:
                    # User Behavior Analysis
                    user_agg = data.groupby('MSISDN/Number').agg({
                        'Dur. (ms)': 'sum',
                        'Total DL (Bytes)': 'sum',
                        'Total UL (Bytes)': 'sum',
                        'Bearer Id': 'count'
                    }).reset_index()

                    user_agg.rename(columns={
                        'Dur. (ms)': 'Total Session Duration (ms)',
                        'Total DL (Bytes)': 'Total Download (Bytes)',
                        'Total UL (Bytes)': 'Total Upload (Bytes)',
                        'Bearer Id': 'Number of Sessions'
                    }, inplace=True)

                    # Plotting Total Session Duration Distribution
                    fig, ax = plt.subplots()
                    sns.histplot(user_agg['Total Session Duration (ms)'], bins=30, kde=True, ax=ax)
                    ax.set_title('Total Session Duration Distribution')
                    st.pyplot(fig)

                    # Bivariate Analysis: Download vs Upload
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='Total Download (Bytes)', y='Total Upload (Bytes)', data=user_agg, ax=ax)
                    ax.set_title('Download vs Upload Data')
                    st.pyplot(fig)

                    # K-Means Clustering
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(user_agg[['Total Session Duration (ms)', 'Total Download (Bytes)', 'Total Upload (Bytes)']])
                    kmeans = KMeans(n_clusters=3)
                    user_agg['Cluster'] = kmeans.fit_predict(scaled_data)

                    # Plotting Cluster Results
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='Total Download (Bytes)', y='Total Upload (Bytes)', hue='Cluster', data=user_agg, palette='Set1', ax=ax)
                    ax.set_title('User Clusters based on Download and Upload')
                    st.pyplot(fig)

                    # Elbow Method Visualization
                    inertia = []
                    for k in range(1, 10):
                        kmeans = KMeans(n_clusters=k)
                        kmeans.fit(scaled_data)
                        inertia.append(kmeans.inertia_)

                    fig, ax = plt.subplots()
                    ax.plot(range(1, 10), inertia)
                    ax.set_title('Elbow Method for Optimal k')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Inertia')
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error executing query: {e}")

elif selected_option == "Sample Data Preview":
    st.header("Display Sample Data and Plot")

    st.subheader("Sample Data Preview")
    sample_data = {
        "Month": ["January", "February", "March", "April", "May", "June"],
        "Sales": [250, 300, 450, 200, 500, 400]
    }
    df = pd.DataFrame(sample_data)
    st.dataframe(df)

    st.subheader("Sample Data Plot")
    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Sales"], marker='o', color='green')
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Sales")
    st.pyplot(fig)

    st.markdown("<h6 style='text-align: center;'>Dashboard Development using Streamlit</h6>", unsafe_allow_html=True)