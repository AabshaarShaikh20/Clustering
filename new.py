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

# Streamlit app title and description
st.title('Clustering App')
st.write('This app uses KMeans clustering to group countries based on their development metrics.')

# Select a cluster to view its characteristics
st.write('Select a cluster to view its characteristics:')
cluster = st.selectbox('Cluster', range(5))

# Show characteristics of the selected cluster
st.write('Cluster', cluster)
st.write('Characteristics:')
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

# Explanation for the cluster colors
st.write('Color Legend:')
st.write('* Cluster 0 (Blue): Strong economic development and high life expectancy')
st.write('* Cluster 1 (Orange): Moderate economic development and medium life expectancy')
st.write('* Cluster 2 (Green): Weak economic development and low life expectancy')
st.write('* Cluster 3 (Red): Unique development profiles')
st.write('* Cluster 4 (Purple): Other development profiles')
