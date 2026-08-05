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
