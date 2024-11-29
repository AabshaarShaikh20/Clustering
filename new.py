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
st.write("Upload your data and observe how the K-Means clustering model assigns data points to different clusters.")

# File upload functionality
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Select features for clustering (ensure the columns selected are numeric)
    st.write("Select the features to use for clustering:")
    features = st.multiselect("Features", options=df.columns.tolist(), default=df.columns.tolist())

    # Check if the user selected any columns
    if len(features) > 0:
        # Preprocess the data
        data = df[features]

        # Standardizing the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

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
    st.write("Please upload a CSV file to get started.")
