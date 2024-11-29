import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the KMeans model
with open('./kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Load and preprocess the dataset
df = pd.read_csv('./World_development_mesurement.csv')
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Streamlit app header
st.title("Clustering Global Development 🌐")
st.markdown("""
Explore global development patterns by clustering countries based on key metrics like GDP, CO2 emissions, and more.  
Dive into insights about economic and environmental trends across clusters!
""")

# Sidebar for filtering and inputs
st.sidebar.header("Filter Options")
show_raw_data = st.sidebar.checkbox("Show Raw Dataset", value=False)
selected_metric = st.sidebar.selectbox(
    "Choose a Metric for Detailed Analysis:",
    df.columns,
    help="Select a feature to view distribution and insights."
)

# Show raw data if selected
if show_raw_data:
    st.subheader("Raw Dataset")
    st.dataframe(df)

# Cluster selection for exploration
cluster_options = list(range(len(set(kmeans.labels_))))
selected_cluster = st.sidebar.selectbox("Select a Cluster to Analyze:", cluster_options)

# Display summary of the selected cluster
st.subheader(f"Summary of Cluster {selected_cluster}")
cluster_data = df[kmeans.labels_ == selected_cluster]
st.write("### Statistical Overview")
st.write(cluster_data.describe())

# Visualize data distribution for the selected metric
st.write(f"### Distribution of '{selected_metric}' Across Clusters")
fig, ax = plt.subplots()
sns.boxplot(x=kmeans.labels_, y=selected_metric, data=df, palette="coolwarm", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel(selected_metric)
ax.set_title(f"Distribution of {selected_metric}")
st.pyplot(fig)

# Scatter plot to show clustering
st.write("### Cluster Visualization")
fig, ax = plt.subplots()
palette = sns.color_palette("Set1", len(cluster_options))
for cluster in cluster_options:
    cluster_subset = df[kmeans.labels_ == cluster]
    ax.scatter(
        cluster_subset.iloc[:, 0], cluster_subset.iloc[:, 1],
        label=f"Cluster {cluster}",
        alpha=0.7,
        s=50
    )
ax.set_title("Clustering of Countries")
ax.set_xlabel("Feature 1: GDP")
ax.set_ylabel("Feature 2: CO2 Emissions")
ax.legend(title="Clusters")
st.pyplot(fig)

# Additional insights
st.write("### Key Takeaways")
cluster_insights = {
    0: "Countries with advanced economies and high sustainability scores.",
    1: "Emerging economies with steady progress in environmental reforms.",
    2: "Developing nations with low industrialization but rising population growth.",
    3: "Resource-rich nations with unique economic profiles.",
    4: "Countries undergoing significant environmental challenges."
}

st.write(f"**Cluster {selected_cluster} Insights:** {cluster_insights.get(selected_cluster, 'No insights available')}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by A Data Enthusiast 💡")
