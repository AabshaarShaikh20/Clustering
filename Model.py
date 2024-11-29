import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained KMeans model
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the dataset
df = pd.read_csv('./World_development_mesurement.csv')

# Clean the data by removing unwanted symbols
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Streamlit app title and description
st.title('🌍 Global Development Clustering App 🌍')
st.write(
    "This app uses KMeans clustering to group countries based on their development metrics, "
    "providing insights into economic and environmental characteristics."
)

# Sidebar with cluster selection
st.sidebar.title("Cluster Selection")
cluster = st.sidebar.selectbox('Select a cluster to view its characteristics:', range(5))

# Display selected cluster characteristics
st.write(f"### Cluster {cluster} Characteristics")
st.write(df[kmeans.labels_ == cluster].describe())

# Enhanced Cluster Color Legend
st.write("### 🌟 Cluster Color Legend 🌟")
st.markdown("""
<style>
.legend-box {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.color-circle {
    width: 15px;
    height: 15px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    border: 1px solid #000;
}
</style>
""", unsafe_allow_html=True)

legend_html = """
<div class="legend-box">
    <div class="color-circle" style="background-color: blue;"></div>
    <span><b>Cluster 1</b>: Strong economic development and high life expectancy</span>
</div>
<div class="legend-box">
    <div class="color-circle" style="background-color: orange;"></div>
    <span><b>Cluster 2</b>: Moderate economic development and medium life expectancy</span>
</div>
<div class="legend-box">
    <div class="color-circle" style="background-color: green;"></div>
    <span><b>Cluster 3</b>: Weak economic development and low life expectancy</span>
</div>
<div class="legend-box">
    <div class="color-circle" style="background-color: red;"></div>
    <span><b>Cluster 4</b>: Unique development profiles</span>
</div>
<div class="legend-box">
    <div class="color-circle" style="background-color: purple;"></div>
    <span><b>Cluster 5</b>: Other development profiles</span>
</div>
"""

st.markdown(legend_html, unsafe_allow_html=True)

# Scatter plot visualization
st.write("### Cluster Scatter Plot")
fig, ax = plt.subplots()
colors = ['blue', 'orange', 'green', 'red', 'purple']
scatter = ax.scatter(
    df.iloc[:, 0], df.iloc[:, 1],
    c=[colors[i] for i in kmeans.labels_],
    alpha=0.6, edgecolor='k'
)
ax.set_xlabel('Feature 1 - GDP')
ax.set_ylabel('Feature 2 - CO2 Emissions')
plt.colorbar(scatter, ax=ax, label="Cluster Labels")
st.pyplot(fig)

# Add a button to update the plot dynamically for the selected cluster
if st.button('Show Selected Cluster Only'):
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df.iloc[kmeans.labels_ == cluster, 0],
        df.iloc[kmeans.labels_ == cluster, 1],
        c=colors[cluster], alpha=0.6, edgecolor='k'
    )
    ax.set_xlabel('Feature 1 - GDP')
    ax.set_ylabel('Feature 2 - CO2 Emissions')
    st.pyplot(fig)

# Display the background image as a watermark
background_image_url = "https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg"
st.markdown(f"""
    <style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-position: center;
    }}
    </style>
""", unsafe_allow_html=True)
