# Setup

This project includes scripts to set up the required Python environment and dependencies, run the Flask backend, and load the Chrome extension manually.

## Prerequisites

- Ubuntu system
- Administrative (sudo) access
- Google Chrome installed (Chrome Version 124.0.6367.203 (Official Build) (64-bit))

## Setup Python Environment

A script is provided to install Python 3.8.10 and all required dependencies for this project. The script is located at `Deepfake-Detection/Scripts/setup.sh`.

### Steps to Run the Setup Script

1. Open a terminal.
2. Navigate to the project directory:
    Deepfake-Detection/Scripts
3. Run the setup script:
   ```sh
   bash setup.sh
   ```

This script will:
- Check if Python 3.8.10 is installed, and install it if necessary.
- Create a virtual environment.
- Install the required packages:
  - pandas==2.0.3
  - scikit-learn==1.2.2
  - numpy==1.23.5
  - tensorflow==2.13.1
  - keras==2.13.1
  - Flask==3.0.2
  - flask-cors==4.0.0
  - requests==2.21.0
  - dlib==19.24.2
  - joblib==1.3.2

## Run the Flask Backend

To run the Flask backend required for the Chrome extension, execute the following script:

### Steps to Run the Flask Script

1. Navigate to the project directory:
    Deepfake-Detection/Scripts
2. Run the setup script:
   ```sh
   bash run_flask_app.sh
   ```

2. The `run_flask_app.sh` script performs the following actions:
   - Checks if the virtual environment exists at Deepfake-Detection/Scripts/myenv. If not, it creates one.
   - Activates the virtual environment.
   - Changes the directory to `Deepfake-Detection/BackendApplication` where Flask application is located.
   - Runs the Flask application using `python3.8`.

## Load the Chrome Extension

### Manual Steps to Load the Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable "Developer mode" using the toggle switch in the top right corner.
3. Click on the "Load unpacked" button.
4. Select the extension’s folder Deepfake-Detection\ChromeExtension

### Notes

- Ensure the Flask backend is running before using the Chrome extension.
- The extension relies on the backend to function correctly.
- Only Scripts, BackendApplication, and ChromeExtension folders are required to run application.
- Machine Learning Folder is provided to show data, implimentation and training workflow of the Machine Learning Models.