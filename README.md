
# Sentence Classifier

## Overview
This project implements a sentence classification system using Streamlit and machine learning embeddings. The application allows users to input sentences and classify them into categories, intensities, and polarities based on pre-defined clusters of sentence embeddings.

## Features
- Generate embeddings for user-input sentences using Ollama's embedding model.
- Classify sentences into categories, intensities, and polarities.
- Visualize classification results with a bar graph.
- Evaluate the model's accuracy using a user-uploaded test dataset.

## Prerequisites
Make sure you have the following installed:
- Python 3.x
- Streamlit
- Ollama
- NumPy
- Pandas
- SciPy
- Matplotlib

You can install the required libraries using pip:
```bash
pip install streamlit ollama numpy pandas scipy matplotlib
```

## Running the Application
To run the application, use the following command in your terminal:
```bash
streamlit run sentence_classifier.py
```

## Usage
1. Open your web browser and navigate to `http://localhost:8501`.
2. Input a sentence in the provided text area and click the "Classify" button to see the classification results.
3. Optionally, upload a CSV file containing a test dataset with `User Input` and `Category` columns to evaluate the model's accuracy.

## Expected CSV Format
The uploaded CSV file must contain the following columns:
- `User Input`: The sentences to classify.
- `Category`: The expected categories for those sentences.

### Example CSV Format:
```csv
User Input,Category
"This is a great product!", Positive
"I had a terrible experience.", Negative
```

## Evaluating Model Accuracy
After uploading a valid CSV file, click the "Evaluate Model" button to see the model's accuracy based on the test dataset. A bar graph will display the predicted versus expected categories.

## Plotting Classification Results
The application will visualize the classification results in a bar graph showing both predicted and expected counts for each category.

## Code Explanation
- **generate_embedding(sentence)**: Generates embeddings for the input sentence using the Ollama model.
- **parse_cluster_data(file_path, Column)**: Parses cluster data from specified text files.
- **find_closest_cluster(cluster_data, target_vector)**: Finds the closest cluster based on both Euclidean distance and cosine similarity.
- **classify_sentence(sentence)**: Classifies a sentence and returns the closest categories for each cluster type.
- **evaluate_model(test_data)**: Evaluates the model's accuracy against a test dataset.
- **plot_classification_results(results)**: Plots the results of the classification in a bar graph.

