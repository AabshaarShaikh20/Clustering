import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained KMeans model from the pickle file
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the dataset
df = pd.read_csv('./World_development_mesurement.csv')

# Clean the data by removing percentage symbols and converting to numeric values
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Set background image using HTML, keeping the layout simple
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg');
            background-size: cover;
            background-position: center;
        }
        .main-container {
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent container */
            border-radius: 10px;
            padding: 30px;
            color: white;
            margin: 20px;
        }
        .sidebar {
            background-color: rgba(0, 0, 0, 0.7);  /* Darker sidebar */
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .header {
            color: white;
            font-size: 28px;
            font-weight: bold;
        }
        .subheader {
            color: white;
            font-size: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and intro inside a transparent container
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<p class="header">Clustering App</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">This app uses KMeans clustering to group countries based on their development metrics.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar content with selection options
st.sidebar.markdown('<div class="sidebar">', unsafe_allow_html=True)
st.sidebar.write("Use the options below to filter and view the results.")
cluster = st.sidebar.selectbox('Select a cluster to view its characteristics:', range(5))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Display characteristics of the selected cluster in the main area
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<p class="subheader">Cluster {cluster}</p>', unsafe_allow_html=True)
st.write(df[kmeans.labels_ == cluster].describe())

# Create a placeholder for the plot
plot_placeholder = st.empty()

# Main scatter plot showing all clusters
with plot_placeholder.container():
    # Create the main scatter plot showing all clusters
    fig, ax = plt.subplots()
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_])
    ax.set_xlabel('Feature 1 - GDP')
    ax.set_ylabel('Feature 2 - CO2 Emissions')
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Focused scatter plot showing only the selected cluster
    st.write(f"Displaying data for Cluster {cluster}")
    df_cluster = df[kmeans.labels_ == cluster]
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], c='black')
    ax.set_xlabel('Feature 1 - GDP')
    ax.set_ylabel('Feature 2 - CO2 Emissions')
    st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

