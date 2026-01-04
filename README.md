# skin-disease-classification
# Skin Disease Classification Using Deep Learning for Automated Dermatological Support

A deep learning–based decision-support system that classifies dermoscopic skin lesion images into **8 diagnostic classes** using a **fine-tuned ResNet-18** model. The system includes **Grad-CAM visualisation** and a **Streamlit web application** with single-image and batch prediction, prediction history, and admin analytics.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only** and is **not** a medical diagnostic device.

---

## ✅ Features

- **8-class skin lesion classification** (ISIC-style)
  - `AK`, `BCC`, `BKL`, `DF`, `MEL`, `NV`, `SCC`, `VASC`
- **ResNet-18 transfer learning** (ImageNet pre-trained)
- **Train / Validation / Test split** using folder-based dataset split
- **Grad-CAM heatmaps** for visual interpretation of model attention
- **Streamlit Web App**
  - Login / Signup
  - Single-image prediction + Grad-CAM
  - Batch prediction (multiple images)
  - Prediction history (user & admin)
  - Admin analytics dashboard
- **SQLite databases**
  - `users.db` (authentication)
  - `history.db` (prediction logs)


Each class folder contains dermoscopic images (jpg/png).

Classes used:
- **AK** = Actinic Keratosis  
- **BCC** = Basal Cell Carcinoma  
- **BKL** = Benign Keratosis  
- **DF** = Dermatofibroma  
- **MEL** = Melanoma  
- **NV** = Melanocytic Nevus  
- **SCC** = Squamous Cell Carcinoma  
- **VASC** = Vascular Lesions  

---

## ⚙️ Installation

### 1) Clone the repository

### 2) Create & activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate

### 3) Install dependencies
pip install streamlit torch torchvision scikit-learn pandas matplotlib pillow tqdm

### 4) 🏋️ Train the Model (Optional)

If you want to train from scratch using your train/val split:

python scripts/train.py

This will:

load data_split/train and data_split/val

train ResNet-18

save checkpoints into checkpoints/

save final model as: skin_model.pth

Training on CPU can be slow. GPU is recommended if available

### 5) ✅ Evaluate on Test Set
python scripts/test_evaluate.py

### 6) Run the Streamlit Web Application
streamlit run app.py






