"""
  File: my_flask_app.py
  Version: 1.0
  Description: This Flask application serves as the backend server for a deepfake detection extension.
  It receives image URLs via POST requests, extracts features from the images,
  predicts the probability of being a deepfake using multiple machine learning models,
  and returns the probability results as a JSON response.
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow import keras
from keras.models import load_model
from image_data_extraction import extract_features_from_image

app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image URL from the request
        image_url = request.json.get('image_url')
        print("Pulled image URL successfuly from user select:", image_url)  # Check what URL is received

        # Call Extract Features Function
        features_dataframe = extract_features_from_image(image_url)
        print("Facial features extraced")

        # Check dimensions of the extracted features
        if features_dataframe.ndim != 2:
            raise ValueError("Extracted features dimension error: Expected 2D array, got {}D".format(features_dataframe.ndim))

        # Load trained models
        print("Loading Models...")
        random_forest_model = pickle.load(open('models/random_forest_extended_ftr.sav', 'rb'))
        svm_model = pickle.load(open('models/svm_model_extended_ftr.sav', 'rb'))
        mlp_model = load_model('models/MLP_best_model_extended_ftr.h5')
        ffnn_model = load_model('models/new_best_model_FFNN_extended_ftr.h5')
        #discriminator = load_model('models/binary_discriminator_GAN_model_extended_ftr_full.h5')
        discriminator = load_model('models/GAN_discriminator_model_extended_ftr_full.h5')
        print("Models Loaded...")

        # Predicting Deepfake probability
        print("Predicting Results...")
        features_array = features_dataframe.to_numpy()  # Convert DataFrame to numpy array for Keras models
        random_forest_prediction = np.squeeze(random_forest_model.predict(features_dataframe)).astype(float)
        svm_prediction = np.squeeze(svm_model.predict(features_dataframe)).astype(float)
        ffnn_prediction = np.squeeze(ffnn_model.predict(features_array)).astype(float)
        mlp_prediction = np.squeeze(mlp_model.predict(features_array)).astype(float)
        gan_prediction = np.squeeze(discriminator.predict(features_array)).astype(float)

        # Predicting Deepfake probability
        print("Random Forest Prediction:", random_forest_prediction)
        print("SVM Prediction:", svm_prediction)
        print("FFNN Prediction:", ffnn_prediction)
        print("MLP Prediction:", mlp_prediction)
        print("GAN Discriminator Prediction:", gan_prediction)

        # Model accuracies
        accuracies = np.array([
            0.7458764685737169,  # Random Forest
            0.7432300446059568,  # SVM
            0.8426,              # MLP
            0.8357,              # FFNN
            0.9999117851257324    # GAN Discriminator
        ])

         # Normalize accuracies to sum to 1
        weights = accuracies / np.sum(accuracies)

        # Weighted average of predictions
        predictions = np.array([
            random_forest_prediction,
            svm_prediction,
            ffnn_prediction,
            mlp_prediction,
            gan_prediction
        ])

        # Weighted average of predictions
        weighted_prediction = np.dot(weights, predictions)

        # Prior probabilities from the dataset
        # 28452 records of DeepFake Images Data
        # 28233 records of Real Images Data
        P_fake = 28452 / (28452 + 28233)
        P_real = 1 - P_fake

        # Posterior probability
        probability_result = weighted_prediction * P_fake / (weighted_prediction * P_fake + (1 - weighted_prediction) * P_real)

        # Convert to percentage for output formatting
        probability_result *= 100
        print("Probability Prediction of DeepFake:", probability_result)

        return jsonify({'probability_results': probability_result}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
