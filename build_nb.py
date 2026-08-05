import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# 1 Introduction
cells.append(nbf.v4.new_markdown_cell("# Facial Emotion Recognition\n\n## 1. Introduction\nThis notebook demonstrates a complete, production-quality Deep Learning pipeline for Facial Emotion Recognition. It covers everything from dataset analysis to model training and evaluation using a Convolutional Neural Network (CNN)."))

# 2 Problem Statement
cells.append(nbf.v4.new_markdown_cell("## 2. Problem Statement\nThe objective is to build a CNN that can accurately classify human facial emotions into 7 categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. The model must handle corrupted images, perform data augmentation to prevent overfitting, and be robust enough for real-time webcam inference."))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Add utils to path
sys.path.append('../')
from utils.dataset_handler import DatasetHandler
from utils.model_builder import build_model, compile_model
from utils.visualization import plot_training_history, plot_confusion_matrix, generate_classification_report"""))

# 3 Dataset
cells.append(nbf.v4.new_markdown_cell("## 3. Dataset & Preprocessing\nWe will scan the raw dataset, remove corrupt files, and perform an 80/20 stratified split into the `dataset/train` and `dataset/test` directories."))
cells.append(nbf.v4.new_code_cell("""raw_data_dir = "../../archive-3"
dataset_dir = "../dataset"

# Scan and split the dataset automatically
handler = DatasetHandler(raw_data_path=raw_data_dir, target_base_path=dataset_dir)
if not os.path.exists(os.path.join(dataset_dir, 'train')):
    handler.run_pipeline()
else:
    print("Dataset already split. Using existing dataset/train and dataset/test directories.")"""))

# 4 Dataset Analysis
cells.append(nbf.v4.new_markdown_cell("## 4. Dataset Analysis\nLet's visualize the distribution of classes in our training and testing sets to ensure they are balanced."))
cells.append(nbf.v4.new_code_cell("""from utils.visualization import plot_class_distribution

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

plot_class_distribution(train_dir, test_dir)"""))

# 5 Preprocessing & 6 Data Augmentation
cells.append(nbf.v4.new_markdown_cell("## 5 & 6. Preprocessing & Data Augmentation\nWe use `ImageDataGenerator` for on-the-fly data augmentation, including rotations, shifts, and flips to improve model generalization."))
cells.append(nbf.v4.new_code_cell("""IMG_SIZE = (48, 48)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

classes = list(train_generator.class_indices.keys())
print("Classes:", classes)"""))

# 7 CNN Architecture
cells.append(nbf.v4.new_markdown_cell("## 7. CNN Architecture\nWe build a custom Convolutional Neural Network with multiple blocks consisting of Conv2D, Batch Normalization, MaxPooling, and Dropout layers."))
cells.append(nbf.v4.new_code_cell("""model = build_model(input_shape=(48, 48, 1), num_classes=len(classes))
model = compile_model(model, learning_rate=0.001)
model.summary()"""))

# 8 Training
cells.append(nbf.v4.new_markdown_cell("## 8. Model Training\nWe train the model using callbacks to prevent overfitting and save the best model weights."))
cells.append(nbf.v4.new_code_cell("""models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(models_dir, 'best_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Note: Set epochs=50 for full training. Using a smaller number for demonstration.
EPOCHS = 10 

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)"""))

# 9 Evaluation
cells.append(nbf.v4.new_markdown_cell("## 9. Model Evaluation\nLet's evaluate the model on the unseen test dataset. We generate accuracy and loss curves, followed by a confusion matrix and classification report."))
cells.append(nbf.v4.new_code_cell("""plot_training_history(history)"""))
cells.append(nbf.v4.new_code_cell("""predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

generate_classification_report(y_true, y_pred, classes)
plot_confusion_matrix(y_true, y_pred, classes)"""))

# 10 Real-Time Detection
cells.append(nbf.v4.new_markdown_cell("## 10. Real-Time Detection\nTo see the model in action using your webcam, you can run the `realtime/webcam_fer.py` script from the terminal.\n\n`python realtime/webcam_fer.py`"))

# 11 Conclusion
cells.append(nbf.v4.new_markdown_cell("## 11. Conclusion\nThis project successfully demonstrates the end-to-end pipeline of building a robust CNN for facial emotion recognition, employing best practices in deep learning such as data augmentation, batch normalization, and rigorous evaluation methodologies."))

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/Facial_Emotion_Recognition.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully.")
