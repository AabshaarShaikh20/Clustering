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

# Set background image using HTML with an overlay for better visibility
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }
        .main {
            background-color: rgba(0, 0, 0, 0.6);  /* Semi-transparent background for text */
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .sidebar {
            background-color: rgba(0, 0, 0, 0.7);  /* Dark background for sidebar */
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app title and description with white text to stand out
st.markdown('<p class="main"><h1>Clustering App</h1></p>', unsafe_allow_html=True)
st.markdown('<p class="main">This app uses KMeans clustering to group countries based on their development metrics.</p>', unsafe_allow_html=True)

# Left Sidebar for user input controls
st.sidebar.markdown('<p class="sidebar"><h3>Controls</h3></p>', unsafe_allow_html=True)
st.sidebar.write("Use the options below to filter and view the results.")

# Cluster selection in the sidebar
cluster = st.sidebar.selectbox('Select a cluster to view its characteristics:', range(5))

# Display characteristics of the selected cluster in the main area
st.markdown('<p class="main"><h3>Cluster {}</h3></p>'.format(cluster), unsafe_allow_html=True)
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

# Explanation for the cluster colors on the sidebar
st.sidebar.markdown('<p class="sidebar"><h4>Cluster Color Legend:</h4></p>', unsafe_allow_html=True)
st.sidebar.write('* Cluster 0 (Blue): Strong economic development and high life expectancy')
st.sidebar.write('* Cluster 1 (Orange): Moderate economic development and medium life expectancy')
st.sidebar.write('* Cluster 2 (Green): Weak economic development and low life expectancy')
st.sidebar.write('* Cluster 3 (Red): Unique development profiles')
st.sidebar.write('* Cluster 4 (Purple): Other development profiles')
