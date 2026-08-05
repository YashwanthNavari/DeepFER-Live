<div align="center">
  <h1>🧠 DeepFER-Live</h1>
  <p><strong>Real-Time Facial Emotion Recognition powered by Deep Learning</strong></p>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</div>

<br/>

## 🌟 Overview

**DeepFER-Live** is a highly optimized, real-time emotion detection AI that analyzes webcam feeds to instantly classify human facial expressions. Built as an internship project, it utilizes a custom, lightweight Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

It can accurately detect 7 core emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

## ✨ Key Features

- ⚡ **Lightweight Architecture**: Designed to train and run incredibly fast on standard CPU hardware without needing an expensive GPU.
- 🎯 **Advanced Pre-processing**: Implements dynamic Haar Cascade bounding box cropping and live histogram equalization to normalize real-world webcam lighting.
- 📊 **Live Probability UI**: Features a custom on-screen graphical bar chart overlay, displaying the model's live confidence scores for all 7 emotions.
- 🎥 **Anti-Flicker Technology**: Utilizes a 5-frame temporal smoothing algorithm (moving average) to provide buttery-smooth and stable prediction tracking.

## 🛠️ Technology Stack

- **Machine Learning**: TensorFlow & Keras
- **Computer Vision**: OpenCV, MediaPipe
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YashwanthNavari/DeepFER-Live.git
   cd DeepFER-Live
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Live Webcam AI**
   ```bash
   python realtime/webcam_fer.py
   ```

## 🧠 Model Training (Optional)
If you wish to retrain the model from scratch, simply run the training pipeline:
```bash
python main_training.py
```
The script will automatically handle data augmentation, early stopping, and save the most accurate weights to `models/best_model.keras`.

<hr/>
<p align="center"><i>Developed by Yashwanth Navari</i></p>
