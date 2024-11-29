import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Custom CSS for background and styling
def add_background():
    st.markdown(
        """
        <style>
        /* Background Image */
        .stApp {
            background: url("https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg" no-repeat center center fixed; 
            background-size: cover;
        }
        /* Text Styling */
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .stSidebar {
            background-color: rgba(0, 0, 0, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background CSS
add_background()

# Load the pre-trained KMeans model
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load and clean the dataset
df = pd.read_csv('./World_development_mesurement.csv')
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Streamlit app title and description
st.title('🌍 Global Development Clustering App 🌍')
st.markdown("""
This app uses **KMeans clustering** to analyze and group countries based on development metrics, 
such as GDP, CO2 emissions, and life expectancy.  
Use this tool to explore patterns in global development!
""")

# Sidebar for user interaction
st.sidebar.header('User Input')
cluster = st.sidebar.selectbox('Select a Cluster:', range(5), format_func=lambda x: f'Cluster {x + 1}')

# Display cluster details
st.subheader(f'Details for Cluster {cluster + 1}')
cluster_data = df[kmeans.labels_ == cluster]
st.write("### Key Characteristics")
st.write(cluster_data.describe())

# Plot cluster data
st.write("### Visualizing Clusters")
fig, ax = plt.subplots()
colors = ['blue', 'orange', 'green', 'red', 'purple']
scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_], alpha=0.7, edgecolor='k')
ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c='black', label=f'Selected Cluster {cluster + 1}', edgecolor='white')
ax.set_xlabel('GDP')
ax.set_ylabel('CO2 Emissions')
ax.legend()
st.pyplot(fig)

# Color legend for clusters
st.write("### Cluster Color Legend")
legend_info = {
    0: 'Strong economic development and high life expectancy',
    1: 'Moderate economic development and medium life expectancy',
    2: 'Weak economic development and low life expectancy',
    3: 'Unique development profiles',
    4: 'Other development profiles'
}

for i, desc in legend_info.items():
    st.write(f'* **Cluster {i + 1}** ({colors[i].capitalize()}): {desc}')
