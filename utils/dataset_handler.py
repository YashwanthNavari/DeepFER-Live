import os
import shutil
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetHandler:
    def __init__(self, raw_data_path, target_base_path):
        self.raw_data_path = raw_data_path
        self.target_base_path = target_base_path
        self.train_dir = os.path.join(self.target_base_path, 'train')
        self.test_dir = os.path.join(self.target_base_path, 'test')
        self.classes = []

    def scan_dataset(self):
        """Scans the existing dataset and collects all valid image paths and their labels."""
        logging.info(f"Scanning dataset in {self.raw_data_path}...")
        image_paths = []
        labels = []
        
        # In our case, the raw dataset has 'train' and 'test' subfolders. We will merge them to do a clean 80/20 split.
        for split in ['train', 'test']:
            split_path = os.path.join(self.raw_data_path, split)
            if not os.path.exists(split_path):
                continue
            
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                if class_name not in self.classes:
                    self.classes.append(class_name)
                
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    
                    # Check if file is corrupt or unreadable
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            image_paths.append(img_path)
                            labels.append(class_name)
                        else:
                            logging.warning(f"Corrupted image found and ignored: {img_path}")
                    except Exception as e:
                        logging.warning(f"Error reading {img_path}: {e}")
                        
        df = pd.DataFrame({'filepath': image_paths, 'label': labels})
        logging.info(f"Total valid images found: {len(df)}")
        return df

    def create_split(self, df, test_size=0.2):
        """Performs a stratified split and copies files to the target directories."""
        logging.info(f"Creating an 80/20 stratified split (Test size: {test_size * 100}%)...")
        
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
        
        self._copy_files(train_df, self.train_dir)
        self._copy_files(test_df, self.test_dir)
        
        logging.info("Dataset split and copying completed successfully.")
        
        # Log distribution
        train_counts = train_df['label'].value_counts()
        test_counts = test_df['label'].value_counts()
        logging.info(f"Train set class distribution:\n{train_counts}")
        logging.info(f"Test set class distribution:\n{test_counts}")
        
    def _copy_files(self, df, target_dir):
        """Helper function to copy files to target directory."""
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        for cls in self.classes:
            os.makedirs(os.path.join(target_dir, cls), exist_ok=True)
            
        for idx, row in df.iterrows():
            src = row['filepath']
            filename = os.path.basename(src)
            # Make a unique filename to prevent overwriting if names overlap in raw data
            unique_filename = f"{idx}_{filename}"
            dst = os.path.join(target_dir, row['label'], unique_filename)
            shutil.copy2(src, dst)
            
    def run_pipeline(self):
        """Executes the full dataset preparation pipeline."""
        df = self.scan_dataset()
        if len(df) == 0:
            logging.error("No images found! Please check the raw data path.")
            return
            
        self.create_split(df, test_size=0.2)

if __name__ == "__main__":
    # Example usage
    raw_data = "../archive-3"
    target_data = "../DeepFER/dataset"
    
    handler = DatasetHandler(raw_data_path=raw_data, target_base_path=target_data)
    handler.run_pipeline()
