import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Custom CSS for background and styling
def add_custom_styles():
    st.markdown(
        """
        <style>
        /* Full app background */
        .stApp {
            background: url("https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #FFD700, #FFEC8B);
            color: black;
            border-radius: 10px;
            padding: 10px;
        }
        /* Sidebar text and headings */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label {
            color: black;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
        }
        /* Main content text styling */
        h1, h2, h3 {
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }
        p, label {
            color: white;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        }
        ul {
            padding-left: 20px;
        }
        li {
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply custom styles
add_custom_styles()

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

# Display details of the selected cluster
st.markdown(f"### Details for Cluster {cluster + 1}")
cluster_data = df[kmeans.labels_ == cluster]
st.write(cluster_data.describe())

# Plot cluster data
st.markdown("### Visualizing Clusters")
fig, ax = plt.subplots()
colors = ['blue', 'orange', 'green', 'red', 'purple']
scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_], alpha=0.7, edgecolor='k')
ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c='black', label=f'Selected Cluster {cluster + 1}', edgecolor='white')
ax.set_xlabel('GDP')
ax.set_ylabel('CO2 Emissions')
ax.legend()
st.pyplot(fig)

# Display the Cluster Color Legend
st.markdown("### Cluster Color Legend")
st.markdown("""
- **Cluster 1 (Blue):** Strong economic development and high life expectancy  
- **Cluster 2 (Orange):** Moderate economic development and medium life expectancy  
- **Cluster 3 (Green):** Weak economic development and low life expectancy  
- **Cluster 4 (Red):** Unique development profiles  
- **Cluster 5 (Purple):** Other development profiles  
""")
