import re
import ollama
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

# Generate embedding for the input sentence
def generate_embedding(sentence):
    embedding = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=sentence
    )
    return np.array(embedding['embedding'])

# Parse cluster data from a file
def parse_cluster_data(file_path, Column):
    clusters = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        current_cluster = None
        for line in lines:
            cluster_match = re.match(rf"{Column}\s*:\s*(.*)", line)
            if cluster_match:
                current_cluster = cluster_match.group(1).strip()
                clusters[current_cluster] = None
            
            embedding_match = re.match(r"Average Embedding:\s*\[(.*)\]", line)
            if embedding_match and current_cluster:
                vector_str = embedding_match.group(1)
                vector = [float(x) for x in vector_str.split(',')]
                clusters[current_cluster] = np.array(vector)
    
    return clusters

# Load cluster data from files
cluster_data_category = parse_cluster_data('final/average_embeddings_by_Category.txt', 'Category')
cluster_data_intensity = parse_cluster_data('final/average_embeddings_by_Intensity.txt', 'Intensity')
cluster_data_polarity = parse_cluster_data('final/average_embeddings_by_Polarity.txt', 'Polarity')

# Find the closest cluster based on both Euclidean distance and cosine similarity
def find_closest_cluster(cluster_data, target_vector):
    best_distance = float('inf')
    best_similarity = float('-inf')
    closest_category = None
    closest_metric = None
    closest_category1 = None
    for category, cluster_vector in cluster_data.items():
        # Calculate Euclidean distance
        distance = euclidean(cluster_vector, target_vector)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(cluster_vector, target_vector)

        # Update if this category is closer based on Euclidean distance
        if distance < best_distance:
            best_distance = distance
            closest_category = category
            closest_metric = ("Euclidean", best_distance)

        # Update if this category is better based on cosine similarity
        if similarity > best_similarity:
            best_similarity = similarity
            closest_category1 = category

    return closest_category, closest_metric, best_similarity, closest_category1

# Classify a sentence based on distance to each cluster
def classify_sentence(sentence):
    target_vector = generate_embedding(sentence)
    
    closest_category, metric, similarity, closest_category1 = find_closest_cluster(cluster_data_category, target_vector)
    closest_intensity, metric1, similarity1, closest_intensity1 = find_closest_cluster(cluster_data_intensity, target_vector)
    closest_polarity, metric2, similarity2, closest_polarity1 = find_closest_cluster(cluster_data_polarity, target_vector)
    
    return {
        'Category': (closest_category, metric),
        'Intensity': (closest_intensity, metric1),
        'Polarity': (closest_polarity, metric2),
        'Category1': (closest_category1, ("Cosine", similarity)),
        'Intensity1': (closest_intensity1, ("Cosine", similarity1)),
        'Polarity1': (closest_polarity1, ("Cosine", similarity2)),
    }

# Evaluate model accuracy
def evaluate_model(test_data):
    correct_predictions = 0
    results = []

    for _, row in test_data.iterrows():
        sentence = row['User Input']
        expected_category = row['Category']
        classification_results = classify_sentence(sentence)
        
        results.append((classification_results['Category'][0], expected_category))
        
        if classification_results['Category'][0] == expected_category:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data) * 100
    return accuracy, results

# Plot classification results
def plot_classification_results(results):
    predicted_categories = [result[0] for result in results]
    expected_categories = [result[1] for result in results]

    # Count occurrences
    predicted_counts = Counter(predicted_categories)
    expected_counts = Counter(expected_categories)

    # Plot predicted categories
    plt.figure(figsize=(10, 5))
    plt.bar(predicted_counts.keys(), predicted_counts.values(), color='blue', alpha=0.7, label='Predicted')
    plt.bar(expected_counts.keys(), expected_counts.values(), color='orange', alpha=0.5, label='Expected', linestyle='dotted')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Classification Results')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Streamlit GUI
st.title("Sentence Classifier")

input_sentence = st.text_area("Enter a sentence to classify:")
if st.button("Classify"):
    if input_sentence.strip():
        classification_results = classify_sentence(input_sentence)
        st.subheader("Classification Results:")
        
        # Display results in a well-formatted table
        results_data = {
            "Cluster Type": [],
            "Closest Category": [],
            "Method": [],
            "Value": []
        }
        
        for cluster_type, (cluster_name, (method, value)) in classification_results.items():
            results_data["Cluster Type"].append(cluster_type)
            results_data["Closest Category"].append(cluster_name)
            results_data["Method"].append(method)
            results_data["Value"].append(f"{value:.4f}")

        # Create a DataFrame and display it as a table
        results_df = pd.DataFrame(results_data)
        st.table(results_df)

    else:
        st.warning("Please enter a sentence before classifying.")

# File uploader for test dataset
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    if 'User Input' in test_data.columns and 'Category' in test_data.columns:
        if st.button("Evaluate Model"):
            accuracy, results = evaluate_model(test_data)  # Evaluate using the uploaded dataset
            st.subheader("Model Accuracy:")
            st.write(f"The model accuracy is: **{accuracy:.2f}%**")

            # Plot classification results
            st.subheader("Classification Results Graph")
            plot_classification_results(results)

    else:
        st.warning("The uploaded file must contain 'User Input' and 'Category' columns.")




