#!/bin/bash

# Function to install Python 3.8.10
install_python() {
  sudo apt update
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-distutils

  # Update alternatives to make python3.8 the default python3
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
  sudo update-alternatives --install /usr/bin/python3.8-config python3-config /usr/bin/python3.8-config 1
}

# Check if Python 3.8.10 is installed
PYTHON_VERSION=$(python3.8 --version 2>&1)
if [[ $PYTHON_VERSION == "Python 3.8.10" ]]; then
  echo "Python 3.8.10 is already installed."
else
  echo "Installing Python 3.8.10..."
  install_python
fi

# Create a virtual environment
python3.8 -m venv ./myenv
source ./myenv/bin/activate

# Install the required packages
pip install --upgrade pip
pip install pandas==2.0.3 scikit-learn==1.2.2 numpy==1.23.5 tensorflow==2.13.1 keras==2.13.1 Flask==3.0.2 flask-cors==4.0.0 requests==2.21.0 dlib==19.24.2 joblib==1.3.2

echo "All required packages have been installed."
