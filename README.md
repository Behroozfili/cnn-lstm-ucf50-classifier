

# 🎥 Video Activity Classification using LRCN (CNN + LSTM) on UCF50 Dataset

**Author:** Behrooz Filzadeh


---

## 📖 Project Overview

This project implements a Long-term Recurrent Convolutional Network (LRCN) to classify human activities from video clips. The model combines CNN layers for spatial feature extraction from individual frames and LSTM layers to capture temporal dynamics across sequences of frames.

The UCF50 dataset is used, containing 50 action classes with realistic user-uploaded videos.

---

## 📋 Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
  - [1. Clone Repository](#1-clone-repository)  
  - [2. Create Virtual Environment (Recommended)](#2-create-virtual-environment-recommended)  
  - [3. Install Dependencies](#3-install-dependencies)  
  - [4. Download and Prepare UCF50 Dataset](#4-download-and-prepare-ucf50-dataset)  
- [Usage](#usage)  
  - [1. Configure Parameters](#1-configure-parameters)  
  - [2. Run the Training Script](#2-run-the-training-script)  
- [Model Architecture (LRCN)](#model-architecture-lrcn)  
- [Results and Evaluation](#results-and-evaluation)  
- [File Description](#file-description)  
- [Customization](#customization)  
- [License](#license)  

---

## ✅ Features

- **Frame Extraction:** Extracts a fixed number of frames (`SEQUENCE_LENGTH`) from each video for consistent input sequences.  
- **Image Preprocessing:** Resizes frames to fixed dimensions (`IMAGE_HEIGHT`, `IMAGE_WIDTH`) and normalizes pixel values.  
- **Selective Class Training:** Allows training on a subset of UCF50 classes (modifiable via `CLASSES_LIST`).  
- **LRCN Model Architecture:**  
  - Uses `TimeDistributed` CNN layers for per-frame spatial feature extraction.  
  - Stacks Conv2D, MaxPooling2D, and Dropout layers for robust feature learning.  
  - LSTM layer to model temporal dependencies in frame sequences.  
  - Dense layers with L2 regularization and Dropout to reduce overfitting.  
- **Training:**  
  - Optimized with Adam optimizer and configurable learning rate.  
  - `EarlyStopping` callback to avoid overfitting and save best model weights.  
- **Evaluation:**  
  - Reports training and validation loss and accuracy per epoch.  
  - Evaluates model on a test set with summary metrics.  
  - Plots training curves for better insight.  
- **Model Persistence:** Saves trained model in HDF5 format (`.h5`).  
- **Reproducibility:** Sets seeds for `numpy`, `random`, and `tensorflow` to ensure consistent results.

---

## 📂 Dataset

The project uses the [UCF50 Dataset](https://www.crcv.ucf.edu/data/UCF50.php), containing 50 action categories and over 6,600 realistic videos sourced from YouTube.

### Dataset Structure

After extraction, the UCF50 folder should look like this:

UCF50/
├── WalkingWithDog/
│ ├── video1.avi
│ └── ...
├── TaiChi/
│ ├── video1.avi
│ └── ...
└── ... (other class folders)

yaml
Copy
Edit

Place the `UCF50` folder in the root directory of this project or update the `DATASET_DIR` variable accordingly.

---

## 🛠 Prerequisites

- Python 3.7 or higher  
- pip package manager  
- Git (optional, for cloning the repo)  
- Adequate disk space to store and extract the UCF50 dataset

---

## ⚙️ Setup

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file with the following content:

nginx
Copy
Edit
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
Then install:

bash
Copy
Edit
pip install -r requirements.txt
Note: The code uses keras from TensorFlow 2.x, so TensorFlow 2.x installation is required.

4. Download and Prepare UCF50 Dataset
Download the dataset from the official website: UCF50 Dataset

Extract the .rar or .zip file to get the UCF50 folder.

Place the UCF50 folder in the project root directory or update DATASET_DIR in the script accordingly.

🚀 Usage
1. Configure Parameters
Open the training script (e.g., train_lrcn.py) and modify parameters if needed:

IMAGE_HEIGHT, IMAGE_WIDTH — Frame size after resizing (default: 64x64)

SEQUENCE_LENGTH — Number of frames per video sequence (default: 20)

DATASET_DIR — Path to the UCF50 dataset folder (default: "UCF50")

CLASSES_LIST — List of class names to train on (default: ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"])

epochs — Number of training epochs (default: 50)

batch_size — Batch size (default: 4)

2. Run the Training Script
bash
Copy
Edit
python train_lrcn.py
The script extracts frames, preprocesses data, builds the model, and starts training.

Training and validation loss/accuracy will be displayed per epoch.

After training, evaluation on the test set is performed, and results are printed.

Training/validation curves are plotted.

The final trained model is saved as cnn_lstm_classification.h5 in the project root.

🏗 Model Architecture (LRCN)
The model architecture consists of:

Input: Sequence of frames with shape (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

TimeDistributed CNN Blocks:

Block 1:

Conv2D (32 filters, 3x3 kernel, ReLU activation, L2 regularization)

MaxPooling2D (2x2)

Dropout (0.3)

Block 2:

Conv2D (64 filters, 3x3 kernel, ReLU, L2 regularization)

MaxPooling2D (2x2)

Dropout (0.3)

Block 3:

Conv2D (128 filters, 3x3 kernel, ReLU, L2 regularization)

MaxPooling2D (2x2)

Dropout (0.3)

TimeDistributed Flatten Layer: Flattens CNN output for each frame.

LSTM Layer:

64 units, L2 regularization, return_sequences=False.

Dense Layers:

Dense (128 units, ReLU, L2 regularization)

Dropout (0.5)

Output Layer:

Dense (number of classes), Softmax activation for multi-class classification.

Compilation:

Optimizer: Adam (learning rate 0.0001)

Loss: Categorical Cross-Entropy

📊 Results and Evaluation
Training Metrics: Training and validation loss/accuracy displayed per epoch.

Test Evaluation: Final accuracy and loss on hold-out test set.

Visualizations: Plots of training and validation loss and accuracy curves.

Model Persistence: Saved model can be loaded for inference or further training.

📂 File Description
train_lrcn.py — Main script for data processing, model definition, training, and evaluation.

requirements.txt — List of Python dependencies.

cnn_lstm_classification.h5 — Saved trained model (generated after training).

UCF50/ — Dataset directory with video files organized by class (user-provided).

🎨 Customization
Dataset:

Modify CLASSES_LIST to select different classes.

Change DATASET_DIR if dataset location differs.

Preprocessing:

Adjust IMAGE_HEIGHT, IMAGE_WIDTH, and SEQUENCE_LENGTH for different input shapes.

Model Architecture:

Tune number of CNN filters, kernel sizes, LSTM units, dropout rates.

Experiment with different recurrent layers (e.g., GRU) or add attention mechanisms.

Training Parameters:

Modify epochs, batch_size, learning rate.

Adjust patience for EarlyStopping callback.

