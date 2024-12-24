import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Access the variables for database connection
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Uncomment below lines for debugging purposes
# print(f'Database Name: {db_name}')
# print(f'User: {db_user}')

def load_data_from_postgres(query):
    """
    Connects to PostgreSQL database and loads data based on the provided query.
    
    Args:
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing the query results, or None in case of an error.
    """
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

        # Load the data into a DataFrame using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_data_using_sqlalchemy(query):
    """
    Connects to PostgreSQL database using SQLAlchemy and loads data based on the provided query.
    
    Args:
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing the query results, or None in case of an error.
    """
    try:
        # Create a connection string for SQLAlchemy
        connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a DataFrame
        df = pd.read_sql_query(query, engine)

        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None