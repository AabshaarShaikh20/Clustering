import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained KMeans model
with open('./kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Load and clean the dataset
df = pd.read_csv('./World_development_mesurement.csv')
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Add cluster labels to the dataframe for better usability
df['Cluster'] = kmeans.labels_

# App title and introduction
st.title("Clustering Global Development 🌍")
st.write("""
Discover development patterns across countries using KMeans clustering.  
Analyze group-specific metrics and visualize trends dynamically!
""")

# Sidebar filters
st.sidebar.header("Explore Clusters")
cluster_selection = st.sidebar.multiselect(
    "Select Clusters to Explore:",
    options=df['Cluster'].unique(),
    default=df['Cluster'].unique(),
    help="Select one or more clusters to focus on specific groups."
)

# Dataset preview toggle
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Dataset Overview")
    st.dataframe(df)

# Filter dataset based on selected clusters
filtered_df = df[df['Cluster'].isin(cluster_selection)]

# Key Metrics
st.subheader("Key Metrics by Cluster")
st.write("The following table summarizes key statistics for each selected cluster.")
st.write(filtered_df.groupby('Cluster').mean().style.background_gradient(cmap="Blues"))

# Scatterplot for cluster visualization
st.subheader("Cluster Visualization")
st.write("This scatterplot highlights the clustering of countries based on selected features.")

# Dropdowns to select features for scatterplot
feature_x = st.selectbox("Select X-Axis Feature", options=df.columns[:-1], index=0)
feature_y = st.selectbox("Select Y-Axis Feature", options=df.columns[:-1], index=1)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=filtered_df[feature_x],
    y=filtered_df[feature_y],
    hue=filtered_df['Cluster'],
    palette="Set2",
    s=50,
    alpha=0.8,
    ax=ax
)
ax.set_title("Scatterplot of Clusters")
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
st.pyplot(fig)

# Cluster-wise distribution plot
st.subheader("Feature Distribution Across Clusters")
selected_feature = st.selectbox(
    "Choose a Feature to View Distribution:",
    options=df.columns[:-1]
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    x='Cluster',
    y=selected_feature,
    data=filtered_df,
    palette="coolwarm",
    ax=ax
)
ax.set_title(f"Distribution of {selected_feature} by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel(selected_feature)
st.pyplot(fig)

# Additional insights and takeaways
st.subheader("Cluster Insights")
for cluster in sorted(cluster_selection):
    st.write(f"### Cluster {cluster}")
    st.write("- **Key Observations**: Describe patterns for this cluster here.")
    st.write("- **Suggestions**: Recommendations for countries in this cluster.")

# Footer
st.sidebar.write("---")
st.sidebar.write("App built for exploring global development trends.")
