import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Custom CSS for background and styling
def add_background():
    st.markdown(
        f"""
        <style>
        /* Background Image */
        .stApp {{
            background: url("https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg") no-repeat center center fixed; 
            background-size: cover;
        }}
        /* Styling for text and boxes */
        h1, h2, h3, h4, h5, h6 {{
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}
        p, label, .stMarkdown {{
            color: white;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        }}
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.7);
        }}
        .info-box {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
        }}
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

# Display cluster details in a styled box
st.markdown(f"""
<div class="info-box">
    <h3>Details for Cluster {cluster + 1}</h3>
    <p>Explore the characteristics of countries in this cluster.</p>
</div>
""", unsafe_allow_html=True)

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

# Display the Cluster Color Legend in a styled box
legend_html = """
<div class="info-box">
    <h3>Cluster Color Legend</h3>
    <ul>
        <li><b>Cluster 1 (Blue):</b> Strong economic development and high life expectancy</li>
        <li><b>Cluster 2 (Orange):</b> Moderate economic development and medium life expectancy</li>
        <li><b>Cluster 3 (Green):</b> Weak economic development and low life expectancy</li>
        <li><b>Cluster 4 (Red):</b> Unique development profiles</li>
        <li><b>Cluster 5 (Purple):</b> Other development profiles</li>
    </ul>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)
