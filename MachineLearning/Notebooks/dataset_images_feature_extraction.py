"""
  File: dataset_images_feature_extraction.py
  Version: 2.0
  Description: This script processes images from the FaceForensics++ dataset, extracting various features for DeepFake detection.
  It utilizes dlib for facial landmark detection, OpenCV for gradient and LBP computations, and FFT for frequency domain features.
  The extracted features are saved to a CSV file for further analysis and model training.
  Features Extracted:
    - Facial landmark distances from the center of the face
    - Gradient magnitudes and orientations for each landmark
    - Local Binary Pattern (LBP) histograms
    - Frequency domain mean and standard deviation
  Author: Roman Barron
  Date: 02/08/2024
  Last Revision: 5/16/2024
"""

#Dependecies
import dlib
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import gc

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor_path = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Prepare DataFrame columns for all features
columns = ["Foldername", "Subfoldername", "Filename"] + ["Landmark {}".format(i) for i in range(1, 69)]
for i in range(1, 69):
    columns += [f"Landmark {i} Gradient Magnitude Max", f"Landmark {i} Gradient Magnitude Mean", f"Landmark {i} Gradient Magnitude Std"]
    columns += [f"Landmark {i} Gradient Orientation Max", f"Landmark {i} Gradient Orientation Mean", f"Landmark {i} Gradient Orientation Std"]
columns += ["LBP Histogram Bin {}".format(i) for i in range(256)]
columns += ["Frequency Domain Mean", "Frequency Domain Std"]

# Initialize DataFrame
df = pd.DataFrame(columns=columns)

# Path to csv file to save dataset
csv_path = "./dataset_extended_gradient_manipulated_rest.csv"
if not os.path.exists(csv_path):
    df.to_csv(csv_path, index=False)

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

# Path to images containing original and manipulated images/frames.
path = "./ExtractedFrames2024/"

# Batch size for writing to CSV
batch_size = 1000  #Saving in batches due to memory constraints
count = 0

# Loop over the image folders recursively to locate .png files
for root, dirs, files in tqdm(os.walk(path)):
    for filename in tqdm(files, desc="Processing files"):
        if filename.endswith(".png"):
            full_path = os.path.join(root, filename)
            img = cv2.imread(full_path)
            if img is None:
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = detector(img_gray, 1)
            if len(faces) == 0:
                continue  # Skip files with no faces

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

            # Folder names for tracking purposes
            foldername = os.path.basename(root)
            subfolder = os.path.basename(os.path.dirname(full_path))

            #Landmark and gradient info extraction
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

                row = [foldername, subfolder, filename] + distances
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

                # Check if it's time to append to Excel
                count += 1
                if count >= batch_size:
                    print("")
                    print("Saving batch to excel csv file...")
                    print("")
                    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                    df = pd.DataFrame(columns=columns)  # Reset DataFrame after writing
                    gc.collect() #invoke garbage collection to free space
                    count = 0


# Write any remaining data to Excel
if not df.empty:
    print("")
    print("Finishing remaining df and saving into to csv file...")
    print("")
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
