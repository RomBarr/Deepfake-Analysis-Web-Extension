Dataset and Models Information

Saved Models Location:
The saved models used for the backend application are located in the following directory:

Deepfake-Detection\BackendApplication\models

Dataset used to train Models:
   - The data used to train the models was initially pulled from the FaceForensics++ dataset, which contains real and manipulated videos. The specific subsets used from FaceForensics++ include original sequences from DeepFakeDetection and YouTube, as well as manipulated sequences from DeepFakeDetection and Face2Face.
   - Frames were extracted from each video to create the dataset. For each image frame, several features were extracted using various resources and techniques:
     - **Facial Landmark Distances**: Using dlib's trained facial landmark detectors, the distances from 68 facial landmarks to the center of the face were extracted.
     - **Gradient Information**: Using OpenCV (cv2), gradient information for each landmark was computed. Specifically, the gradient magnitude and orientation were calculated using Sobel operators, and statistics (maximum, mean, and standard deviation) were derived for each landmark.
     - **Local Binary Patterns (LBP)**: Histograms of Local Binary Patterns were computed to capture texture information from the images. 
     - **Frequency Domain Features**: Mean and standard deviation were computed from the frequency domain representation of the images
   - After feature extraction, the dataset underwent standardization and dimensionality reduction:
     - **Standardization**: The features were standardized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.
     - **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the dimensionality of the dataset, retaining 90% of the variance while reducing the number of features.
   - The preprocessed dataset, including the transformed features and the target labels, was saved to a CSV file for training the machine learning models. Additionally, the PCA loadings and scaler were saved to transform new testing data consistently.
   - The saved CSV file is referenced as the 'transformed' dataset.


Implemented Models:

1. Support Vector Machine (SVM):
   - Description: This model is a Support Vector Machine (SVM) classifier used to detect DeepFake images. The model was trained using the 'transformed' dataset where features were prepared from the dataset, excluding target feature. The dataset was split into training and testing sets, and a GridSearchCV with StratifiedKFold cross-validation was used to find optimal hyperparameters for the SVM. The model was evaluated using accuracy as the scoring metric. The best parameters found were used to train the final SVM model, which was then saved for future use.


2. Random Forest (RF):
   - Description: This model is a Random Forest classifier used to detect DeepFake images. The model was trained using 'transformed' dataset, where features were prepared from the dataset, excluding the target feature. The dataset was split into training and testing sets. Cross-validation was performed to check for overfitting and to evaluate the model's performance. The Random Forest classifier was then trained and tested on training data testing data respectively. The model's accuracy, classification report, and confusion matrix were printed to evaluate its performance. The trained model was saved for future use.


3. Feedforward Neural Network (FFNN):
   - Description: This model is a Feedforward Neural Network (FFNN) used to detect DeepFake images. The model was trained using 'transformed' dataset, where features were prepared from the dataset, excluding the target feature. The dataset was split into training and testing sets using K-fold cross-validation with 5 folds.

For each fold, the FFNN architecture was re-initialized and trained on the training data with early stopping and learning rate reduction implemented to avoid overfitting. Model checkpoints were used to save the best model based on validation accuracy. The model was compiled using the Adam optimizer and binary cross-entropy loss. The best model was saved validated by cross-validation results, including accuracy and loss for each fold.

4. Multilayer Perceptron (MLP):
   - Description: This model is a Multilayer Perceptron (MLP) used to detect DeepFake images. The model was trained using 'transformed' dataset, where features were prepared from the dataset, excluding the target feature. The dataset was split into training and testing sets. The model was compiled using the Adam optimizer with a learning rate of 0.0001 and binary cross-entropy loss. Early stopping, model checkpointing, and learning rate reduction were implemented as callbacks to prevent overfitting and improve training.

The MLP model was trained on the training data and validated on a validation split. The best model based on validation loss was saved, and the final model was evaluated on the test data, with accuracy and loss metrics printed to assess performance.

The MLP model differs from the FFNN in its use of Batch Normalization layers and additional hidden layers with different dropout rates, leading to potentially more stable and optimal training.


5. Generative Adversarial Network (GAN) Discriminator:
   - This model is a Generative Adversarial Network (GAN) discriminator used to detect DeepFake images. The GAN consists of two parts: a generator and a discriminator. The discriminator's role is to distinguish between real and generated (fake) data, while the generator creates fake data to fool the discriminator.

The model was compiled using the Adam optimizer with a reduced learning rate to ensure stability during training and binary cross-entropy loss. The discriminator was trained in conjunction with the generator using a custom training loop that updates both models iteratively.

The generator creates fake data samples, and the discriminator learns to distinguish between real and fake samples. The discriminator's performance is evaluated by its ability to correctly classify real and fake data, with accuracy and loss metrics recorded.