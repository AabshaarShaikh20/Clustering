import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Custom CSS for background and styling
def add_custom_styles():
    st.markdown(
        """
        <style>
        /* Full app background */
        .stApp {
            background: linear-gradient(135deg, #FFEC8B, #FFD700); /* Soft golden gradient */
            background-size: cover;
            color: #333333;  /* Dark text for good contrast */
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #FFDA44, #FFB300);  /* Golden gradient */
            color: #5c3a21;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Sidebar headers and labels */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label {
            font-family: 'Arial', sans-serif;
            color: #5c3a21;
            font-weight: bold;
            font-size: 22px;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.7);
        }

        /* Sidebar buttons and dropdowns */
        section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stButton {
            background-color: #FFDA44;
            color: #5c3a21;
            font-weight: bold;
            border-radius: 12px;
            margin-top: 10px;
        }

        /* Hover effect on sidebar buttons */
        section[data-testid="stSidebar"] .stButton:hover {
            background-color: #FFB300;
            color: white;
        }

        /* Main content text styling */
        h1, h2, h3 {
            color: #5c3a21;
            text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.7);
        }
        p, label, ul, li {
            color: #333333;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        }
        li {
            font-size: 16px;
        }

        /* Color the plot elements to match the golden theme */
        .matplotlib {
            background-color: #FFEC8B;
            color: #5c3a21;
        }

        /* Styling for the color legend and table */
        .color-legend {
            background-color: #FFEC8B;
            border-radius: 12px;
            padding: 15px;
            font-weight: bold;
            color: #5c3a21;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Load pre-trained KMeans model
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the dataset
df = pd.read_csv('./World_development_mesurement.csv')

# Clean the data (remove '%' symbols and convert to numeric)
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Set up Streamlit app
st.title('Clustering Global Development App')
st.write('This app uses KMeans clustering to group countries based on their development metrics.')

# Sidebar for cluster selection
st.sidebar.header('Select Cluster')
cluster = st.sidebar.selectbox('Cluster', range(5))

# Display cluster characteristics
st.write('Cluster', cluster)
st.write('Characteristics:')
st.write(df[kmeans.labels_ == cluster].describe())

# Plotting the scatter plot
st.write("### Country Development Scatter Plot")

# Create a plot and display it dynamically based on cluster selection
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red', 'purple']
scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_])
ax.set_xlabel('Feature 1 - GDP')
ax.set_ylabel('Feature 2 - CO2 Emissions')
plt.colorbar(scatter)
st.pyplot(fig)

# Display a color legend for clusters
st.write("### Cluster Color Legend")
st.markdown(
    """
    <div class="color-legend">
    **Cluster 1 (Blue):** Strong economic development and high life expectancy<br>
    **Cluster 2 (Orange):** Moderate economic development and medium life expectancy<br>
    **Cluster 3 (Green):** Weak economic development and low life expectancy<br>
    **Cluster 4 (Red):** Unique development profiles<br>
    **Cluster 5 (Purple):** Other development profiles
    </div>
    """, unsafe_allow_html=True
)

# Add custom styles to make the app look polished
add_custom_styles()
