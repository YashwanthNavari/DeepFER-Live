import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model

from utils.dataset_handler import DatasetHandler
from utils.model_builder import build_model, compile_model
from utils.visualization import plot_training_history, plot_confusion_matrix, generate_classification_report

def main():
    parser = argparse.ArgumentParser(description="Train Facial Expression Recognition Model")
    parser.add_argument('--resume', action='store_true', help="Resume training from the saved best model")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    args = parser.parse_args()

    # Paths
    raw_data_dir = "../archive-3"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_dir, 'dataset')
    models_dir = os.path.join(project_dir, 'models')
    outputs_dir = os.path.join(project_dir, 'outputs')
    graphs_dir = os.path.join(outputs_dir, 'graphs')
    cm_dir = os.path.join(outputs_dir, 'confusion_matrix')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    # 1. Dataset Preparation
    print("--- Phase 1 & 2: Dataset Setup & Analysis ---")
    handler = DatasetHandler(raw_data_path=raw_data_dir, target_base_path=dataset_dir)
    # If the train and test directory don't exist inside DeepFER, run pipeline
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        handler.run_pipeline()
    else:
        print("Dataset already split. Skipping split phase.")

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    # 2. Preprocessing & Data Augmentation
    print("--- Phase 4 & 5: Preprocessing & Data Augmentation (tf.data) ---")
    IMG_SIZE = (48, 48)
    BATCH_SIZE = 64

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    classes = train_dataset.class_names
    
    # Data Augmentation & Normalization Pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode='nearest'),
    ])
    
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_train(x, y):
        x = x / 255.0
        x = data_augmentation(x, training=True)
        return x, y

    def prepare_test(x, y):
        x = x / 255.0
        return x, y
        
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.map(prepare_train, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    
    test_dataset = test_dataset.map(prepare_test, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

    # 3. Model Building
    print("--- Phase 7: CNN Model ---")
    best_model_path = os.path.join(models_dir, 'best_model.keras')
    
    if args.resume and os.path.exists(best_model_path):
        print(f"Resuming training from {best_model_path}")
        model = load_model(best_model_path)
    else:
        if args.resume:
            print(f"Warning: --resume flag passed but no saved model found at {best_model_path}. Starting from scratch.")
        model = build_model(input_shape=(48, 48, 1), num_classes=len(classes))
        model = compile_model(model, learning_rate=0.001)
        
    model.summary()

    # 4. Training
    print("--- Phase 8: Model Training ---")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 5. Evaluation & Outputs
    print("--- Phase 9 & 12: Model Evaluation & Outputs ---")
    plot_training_history(history, save_dir=graphs_dir)
    
    # Predict on test set
    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions, axis=1)
    
    y_true_one_hot = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=1)
    
    generate_classification_report(y_true, y_pred, classes, save_dir=cm_dir)
    plot_confusion_matrix(y_true, y_pred, classes, save_dir=cm_dir)
    
    print("--- Project Execution Completed! ---")

if __name__ == "__main__":
    main()
