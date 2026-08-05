<div align="center">
  
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=35&pause=1000&color=00FF99&center=true&vCenter=true&width=800&lines=DeepFER-Live;Real-Time+Emotion+AI;Powered+by+Deep+Learning" alt="Typing SVG" />
  
  <img src="banner.png" width="100%" alt="DeepFER Banner" style="border-radius:15px; margin: 15px 0;" />

  <p><strong>Advanced Facial Emotion Recognition System</strong></p>

  <p>
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
    <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" alt="Keras" />
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </p>
  
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />
</div>

## 🌌 Overview

**DeepFER-Live** is a state-of-the-art, highly optimized Artificial Intelligence designed to analyze webcam feeds and classify human facial expressions in **real-time**. Built from the ground up as a specialized internship project, it leverages a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset to instantly detect the nuances of human emotion.

### Detects 7 Core Emotions:
`😡 Angry` | `🤢 Disgust` | `😨 Fear` | `😄 Happy` | `😐 Neutral` | `😢 Sad` | `😲 Surprise`

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />

## ✨ Premium Features

<table>
  <tr>
    <td><img src="https://cdn-icons-png.flaticon.com/512/2115/2115951.png" width="40"/></td>
    <td><strong>Live Probability UI</strong><br/>Custom on-screen graphical bar chart overlay, displaying the model's live confidence scores dynamically.</td>
    <td><img src="https://cdn-icons-png.flaticon.com/512/10490/10490218.png" width="40"/></td>
    <td><strong>Anti-Flicker Technology</strong><br/>Utilizes a 5-frame temporal smoothing algorithm (moving average) for buttery-smooth tracking.</td>
  </tr>
  <tr>
    <td><img src="https://cdn-icons-png.flaticon.com/512/2103/2103131.png" width="40"/></td>
    <td><strong>Lightweight Architecture</strong><br/>Engineered to train and run incredibly fast on standard CPU hardware without needing an expensive GPU.</td>
    <td><img src="https://cdn-icons-png.flaticon.com/512/2083/2083213.png" width="40"/></td>
    <td><strong>Smart Pre-processing</strong><br/>Implements dynamic Haar Cascade bounding box cropping and live histogram equalization.</td>
  </tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />

## 🔬 How It Works (The Technical Pipeline)

DeepFER-Live operates on a highly optimized, 4-step computer vision pipeline that processes frames in under `50ms` to achieve real-time speed on consumer hardware:

1. **Face Detection (Haar Cascade / MediaPipe)**: The webcam grabs a live video frame and scans it using a localized bounding-box algorithm to isolate human faces from the background.
2. **Pre-processing (Grayscale & Equalization)**: The cropped face is converted to grayscale to remove unnecessary color channels. Histogram equalization is then applied to normalize shadows and harsh lighting.
3. **Dimensionality Reduction**: The image is aggressively down-scaled to `48x48 pixels` to match the exact input dimensions of the neural network.
4. **CNN Inference**: The processed array is passed through the pre-trained DeepFER Convolutional Neural Network. The model outputs a Softmax array of 7 probabilities, which is temporally smoothed and drawn to the UI.

## 🧠 Model Architecture & Dataset

This model was trained from scratch using the highly respected **FER-2013 Dataset** (originally introduced in the ICML 2013 Challenges in Representation Learning). The dataset consists of `28,709` training images and `3,589` validation images.

To prevent overfitting and maximize inference speed on standard CPUs, the custom architecture utilizes:
- Multiple staggered **Conv2D** layers for spatial feature extraction (edges, curves, micro-expressions).
- **MaxPooling2D** layers to reduce spatial dimensions and computation load.
- **Dropout** layers randomly deactivating neurons during training to enforce robust learning.
- A final **Dense (Fully Connected)** layer with a `Softmax` activation function to output clear probability confidence scores.
- Advanced **Data Augmentation** during training (random zooming, horizontal flipping, and brightness adjustment) to ensure the model generalizes perfectly to real-world webcam feeds.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />

## 📂 Project Structure

```text
DeepFER-Live/
├── dataset/                  # Contains raw & processed FER2013 images
├── models/
│   └── best_model.keras      # Pre-trained Lightweight CNN weights
├── realtime/
│   ├── webcam_fer.py         # Main script for live AI detection (Run this!)
│   └── haarcascade...xml     # OpenCV fallback detection model
├── utils/
│   ├── dataset_handler.py    # Data pipeline and augmentation logic
│   ├── model_builder.py      # CNN Architecture definition
│   └── visualization.py      # Plotting curves and confusion matrices
├── main_training.py          # Script to train the model from scratch
└── requirements.txt          # Python dependencies
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/YashwanthNavari/DeepFER-Live.git
cd DeepFER-Live
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the AI!**
```bash
python realtime/webcam_fer.py
```
*(Press `q` to safely exit the webcam window)*

---

## 🧠 Model Training (Optional)
If you wish to retrain the neural network from scratch using your own parameters, simply run the training pipeline:
```bash
python main_training.py --epochs 50
```
The script will automatically handle data augmentation, execute early stopping, and save the most accurate weights to the `models/` directory.

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" />
  <p><i>Developed with ❤️ by Yashwanth Navari</i></p>
</div>
