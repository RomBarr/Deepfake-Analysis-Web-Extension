"""
  File: image_data_extraction.py
  Version: 1.0
  Description:
    This script downloads an image from a given URL and extracts various features for DeepFake detection.
    It uses dlib for facial landmark detection, OpenCV for gradient and LBP computations, and FFT for frequency domain features.
    The extracted features are then standardized and reduced using pre-saved StandardScaler and PCA models.
  Features Extracted:
    - Facial landmark distances from the center of the face
    - Gradient magnitudes and orientations for each landmark
    - Local Binary Pattern (LBP) histograms
    - Frequency domain mean and standard deviation
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
"""

import requests
import dlib
import cv2
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the pre-trained facial landmark detector
predictor_path = "artifacts/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to compute LBP histogram
def compute_lbp_histogram(img_gray):
    lbp = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    lbp = lbp / np.sum(lbp)  # Normalize the histogram
    return lbp.flatten()

# Function to calculate statistics for gradients
def calculate_gradient_stats(magnitude, angle):
    mag_max = np.max(magnitude)
    mag_mean = np.mean(magnitude)
    mag_std = np.std(magnitude)
    ang_max = np.max(angle)
    ang_mean = np.mean(angle)
    ang_std = np.std(angle)
    return mag_max, mag_mean, mag_std, ang_max, ang_mean, ang_std

def extract_features_from_image(image_url):
    print("Received image URL in image_preprocess_function:", image_url)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()  # Raises a HTTPError for bad responses
        print("Image processed successfully")
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        logging.error(f"HTTP Error: {errh}")
        return jsonify({"error": f"HTTP Error: {str(errh)}"}), 500
    except Exception as e:
        print(f"An internal error occurred: {e}")
        logging.exception("An error occurred: ")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

    if response.status_code != 200:
        raise ValueError(f"Failed to download image from {image_url}, status code: {response.status_code}")

    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    if img_array.size == 0:
        raise ValueError("No data in image array, possibly due to a bad download.")

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image, the data might be corrupted or in an unsupported format.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(img, 1)
    if len(faces) == 0:
        raise ValueError("No recognizable facial features found in the image")

    print("Preparing Dataframe columns...")
    # Prepare DataFrame columns for all features
    columns = ["Landmark {}".format(i) for i in range(1, 69)]
    for i in range(1, 69):
        columns += [f"Landmark {i} Gradient Magnitude Max", f"Landmark {i} Gradient Magnitude Mean", f"Landmark {i} Gradient Magnitude Std"]
        columns += [f"Landmark {i} Gradient Orientation Max", f"Landmark {i} Gradient Orientation Mean", f"Landmark {i} Gradient Orientation Std"]
    columns += ["LBP Histogram Bin {}".format(i) for i in range(256)]
    columns += ["Frequency Domain Mean", "Frequency Domain Std"]

    # Initialize DataFrame
    df = pd.DataFrame(columns=columns)

    # Compute the Local Binary Pattern (LBP) histogram for the grayscale image
    lbp_histogram = compute_lbp_histogram(img_gray)
    # Perform a 2D Fast Fourier Transform (FFT) on the grayscale image
    f_transform = np.fft.fft2(img_gray)
    # Calculate the magnitude spectrum of the FFT
    magnitude_spectrum = np.abs(f_transform)
    # Compute the mean of the magnitude spectrum
    freq_mean = np.mean(magnitude_spectrum)
    # Compute the standard deviation of the magnitude spectrum
    freq_std = np.std(magnitude_spectrum)

    #Extracting landmark and gradient info
    for face in faces:
        landmarks = predictor(img_gray, face)

        x_center = int((face.left() + face.right()) / 2)
        y_center = int((face.top() + face.bottom()) / 2)

        distances = []
        gradient_magnitude_maxs = []
        gradient_magnitude_means = []
        gradient_magnitude_stds = []
        gradient_orientation_maxs = []
        gradient_orientation_means = []
        gradient_orientation_stds = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            distances.append(distance)

            if 0 <= x-10 and x+10 < img.shape[1] and y-10 >= 0 and y+10 < img.shape[0]:
                patch = img[y-10:y+10, x-10:x+10]
                sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=5)
                magnitude, angle = cv2.cartToPolar(sobelx, sobely)
                mag_max, mag_mean, mag_std, ang_max, ang_mean, ang_std = calculate_gradient_stats(magnitude, angle)
                gradient_magnitude_maxs.append(mag_max)
                gradient_magnitude_means.append(mag_mean)
                gradient_magnitude_stds.append(mag_std)
                gradient_orientation_maxs.append(ang_max)
                gradient_orientation_means.append(ang_mean)
                gradient_orientation_stds.append(ang_std)

        row = distances
        row += gradient_magnitude_maxs + gradient_magnitude_means + gradient_magnitude_stds
        row += gradient_orientation_maxs + gradient_orientation_means + gradient_orientation_stds
        row += list(lbp_histogram) + [freq_mean, freq_std]

        if len(row) == len(df.columns):
            df.loc[len(df)] = row
        else:
            print("")
            print("Expected columns:", len(df.columns))
            print("Row length:", len(row))
            print(f"Distances length: {len(distances)}")
            print(f"Gradient magnitude maxs length: {len(gradient_magnitude_maxs)}")
            print(f"Gradient magnitude means length: {len(gradient_magnitude_means)}")
            print(f"Gradient magnitude stds length: {len(gradient_magnitude_stds)}")
            print(f"Gradient orientation maxs length: {len(gradient_orientation_maxs)}")
            print(f"Gradient orientation means length: {len(gradient_orientation_means)}")
            print(f"Gradient orientation stds length: {len(gradient_orientation_stds)}")
            print(f"LBP histogram length: {len(lbp_histogram)}")
            print(f"Frequency stats length: 2")
            print("Mismatch in row and column length; row not added.")
            print("Press Enter to continue...")
            print("")

    print("DataFrame contents before post-processing:")
    print(df.columns)
    print(df)

    try:
        # Load the saved StandardScaler object and transform the concatenated features
        scaler = load('./artifacts/scaler.joblib')
        ftr_scaled = scaler.transform(df)
        app.logger.info("DataFrame contents after StandardScaler transformation successful.")
    except Exception as e:
        app.logger.error("Failed to load or transform with StandardScaler:", exc_info=True)
        return jsonify({'error': f"Scaler error: {str(e)}"}), 500

    try:
        # Load the saved StandardScaler object and transform the concatenated features
        pca = load('./artifacts/pca.joblib')
        ftr_pca = pca.transform(ftr_scaled)
        app.logger.info("DataFrame contents after PCA transformation successful.")
    except Exception as e:
        app.logger.error("Failed to load or transform with PCA:", exc_info=True)
        return jsonify({'error': f"Scaler error: {str(e)}"}), 500

    print("Image Data Extraction Completed")
    print(ftr_pca)
    print("Number of PCA components:", pca.n_components_)

    # Convert the PCA output back to a DataFrame (with column names for components)
    df_pca = pd.DataFrame(ftr_pca, columns=[f'Principal Component {i+1}' for i in range(ftr_pca.shape[1])])

    return df_pca
