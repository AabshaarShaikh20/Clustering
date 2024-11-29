import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
        /* Text Styling */
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
        .css-1p1n3ar {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background CSS
add_background()

# Streamlit Title and Description
st.title("K-Means Clustering Deployment")
st.write("Explore K-Means clustering on a predefined dataset.")

# Load the pre-trained KMeans model
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Predefined Data (Example)
# You can replace this with any static data
data = {
    'Feature_1': [1.2, 3.4, 2.2, 4.5, 5.0, 6.7, 7.8, 3.3],
    'Feature_2': [3.1, 2.3, 5.1, 4.3, 1.0, 3.3, 2.0, 7.4],
    'Feature_3': [2.5, 3.0, 4.5, 5.1, 2.9, 3.2, 3.3, 2.4]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Show the static data to the user
st.write("Data Preview:")
st.dataframe(df)

# Select features for clustering (ensure the columns selected are numeric)
st.write("Select the features to use for clustering:")
features = st.multiselect("Features", options=df.columns.tolist(), default=df.columns.tolist())

# Check if the user selected any columns
if len(features) > 0:
    # Preprocess the data
    selected_data = df[features]

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    # Make predictions using the pre-trained K-Means model
    clusters = kmeans.predict(scaled_data)

    # Add the predicted clusters to the dataframe
    df['Cluster'] = clusters

    # Display the results with clusters
    st.write("Clustering Results:")
    st.dataframe(df)

    # Plot the clusters
    st.write("Cluster Visualization:")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='viridis', s=100, edgecolor='black')
    plt.title('K-Means Clustering Results')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    st.pyplot()

else:
    st.write("Please select the features to use for clustering.")

