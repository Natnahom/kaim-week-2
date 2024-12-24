import pandas as pd
from sqlalchemy import create_engine

def execute_telecom_queries(db_url):
    # Create a database engine using the provided URL
    engine = create_engine(db_url)

    # 1. Count of unique IMSIs
    unique_imsi_count = pd.read_sql_query(
        """
        SELECT COUNT(DISTINCT "IMSI") AS unique_imsi_count
        FROM xdr_data;
        """, engine
    )

    # 2. Average duration of calls (in milliseconds)
    average_duration = pd.read_sql_query(
        """
        SELECT AVG("Dur. (ms)") AS average_duration
        FROM xdr_data
        WHERE "Dur. (ms)" IS NOT NULL;
        """, engine
    )

    # 3. Total Data usage per user (upload and download)
    total_data_usage = pd.read_sql_query(
        """
        SELECT "IMSI",
               SUM("Total UL (Bytes)") AS total_ul_bytes,
               SUM("Total DL (Bytes)") AS total_dl_bytes
        FROM xdr_data
        GROUP BY "IMSI"
        ORDER BY total_dl_bytes DESC
        LIMIT 10;
        """, engine
    )

    # 4. Average RTT (Round Trip Time) by Last Location Name
    avg_rtt_by_location = pd.read_sql_query(
        """
        SELECT "Last Location Name",
               AVG("Avg RTT DL (ms)") AS avg_rtt_dl
        FROM xdr_data
        GROUP BY "Last Location Name"
        HAVING COUNT(*) > 10
        ORDER BY avg_rtt_dl DESC;
        """, engine
    )

    # Return results as a dictionary for easy access
    return {
        "unique_imsi_count": unique_imsi_count,
        "average_duration": average_duration,
        "total_data_usage": total_data_usage,
        "avg_rtt_by_location": avg_rtt_by_location,
    }        

    