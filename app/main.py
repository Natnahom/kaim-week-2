import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple Dashboard", layout="wide")

with st.sidebar:
    selected_option = st.radio("Select an Option:", ["Upload CSV Preview", "Sample Data Preview", "Task Visualizations"])

st.title("Simple Streamlit Dashboard")

if selected_option == "Upload CSV Preview":
    st.header("Option 1: Upload CSV and Plot Data")

    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        st.subheader("Graph of Uploaded Data")

        if st.checkbox("Show Plot"):
            if data.shape[1] >= 2:
                x_col = st.selectbox("Select X-axis column", data.columns)
                y_col = st.selectbox("Select Y-axis column", data.columns)

                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(data[x_col], data[y_col], marker='o', color='b')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
                st.pyplot(fig)
            else:
                st.warning("Uploaded data must have at least two columns for plotting.")
    else:
        st.warning("Please upload a CSV file to display and plot the data.")

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
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(df["Month"], df["Sales"], marker='o', color='green')
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Sales")
    st.pyplot(fig)

    st.markdown("<h6 style='text-align: center;'>Dashboard Development using Streamlit</h6>", unsafe_allow_html=True)

elif selected_option == "Task Visualizations":
    st.header("Visualizations for Tasks 1, 2, 3, and 4")

    uploaded_file = st.file_uploader("Upload CSV for Task Visualizations", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Task 1: Average RTT
        st.subheader("Task 1: Average RTT")
        avg_rtt_data = data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean()
        st.bar_chart(avg_rtt_data)

        # Task 2: Average Bearer Throughput
        st.subheader("Task 2: Average Bearer Throughput")
        avg_throughput_data = data[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean()
        st.bar_chart(avg_throughput_data)

        # Task 3: TCP Retransmissions
        st.subheader("Task 3: TCP Retransmissions")
        tcp_retrans_data = data[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean()
        st.bar_chart(tcp_retrans_data)

        # Task 4: Data Usage by Application
        st.subheader("Task 4: Data Usage by Application")
        app_data = {
            'Social Media': data['Social Media DL (Bytes)'].sum() + data['Social Media UL (Bytes)'].sum(),
            'Google': data['Google DL (Bytes)'].sum() + data['Google UL (Bytes)'].sum(),
            'Email': data['Email DL (Bytes)'].sum() + data['Email UL (Bytes)'].sum(),
            'YouTube': data['Youtube DL (Bytes)'].sum() + data['Youtube UL (Bytes)'].sum(),
            'Netflix': data['Netflix DL (Bytes)'].sum() + data['Netflix UL (Bytes)'].sum(),
            'Gaming': data['Gaming DL (Bytes)'].sum() + data['Gaming UL (Bytes)'].sum(),
            'Other': data['Other DL (Bytes)'].sum() + data['Other UL (Bytes)'].sum(),
        }
        st.bar_chart(app_data)

    else:
        st.warning("Please upload a CSV file to visualize tasks.")

    st.markdown("<h6 style='text-align: center;'>Task Visualizations</h6>", unsafe_allow_html=True)