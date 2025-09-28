"""
Data preprocessing utilities for Egyptian ID dataset preparation and YOLO training.

This module provides classes and functions for:
- Dataset downloading and organization
- Annotation processing and visualization
- Dataset splitting for training/validation/test
- YOLO format preparation
"""

import os
import shutil
import random
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles dataset downloading from Kaggle and file organization."""
    
    def __init__(self, dataset_identifier: str):
        """
        Initialize dataset downloader.
        
        Args:
            dataset_identifier: Kaggle dataset identifier (e.g., "username/dataset-name")
        """
        self.dataset_identifier = dataset_identifier
        self.download_path = None
    
    def download_dataset(self) -> str:
        """
        Download dataset from Kaggle.
        
        Returns:
            Path to downloaded dataset
            
        Raises:
            Exception: If download fails
        """
        try:
            logger.info(f"Downloading dataset: {self.dataset_identifier}")
            self.download_path = kagglehub.dataset_download(self.dataset_identifier)
            logger.info(f"Dataset downloaded to: {self.download_path}")
            return self.download_path
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    @staticmethod
    def copy_files_to_destination(source_folders: List[str], destination_folder: str) -> None:
        """
        Copy files from multiple source folders to a destination folder.
        
        Args:
            source_folders: List of source folder paths
            destination_folder: Destination folder path
        """
        # Create destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        copied_count = 0
        for folder in source_folders:
            if not os.path.exists(folder):
                logger.warning(f"Source folder not found: {folder}")
                continue
                
            folder_files = os.listdir(folder)
            for filename in folder_files:
                source_path = os.path.join(folder, filename)
                destination_path = os.path.join(destination_folder, filename)
                
                try:
                    shutil.copy2(source_path, destination_path)
                    copied_count += 1
                except Exception as e:
                    logger.error(f"Failed to copy {filename}: {e}")
            
            logger.info(f"Copied files from {folder} to {destination_folder}")
        
        logger.info(f"Total files copied: {copied_count}")


class AnnotationProcessor:
    """Handles annotation processing, visualization, and format conversion."""
    
    @staticmethod
    def visualize_polygon(image_path: str, label_path: str) -> Optional[np.ndarray]:
        """
        Load image and draw polygon annotations from YOLO format label file.
        
        Args:
            image_path: Path to image file
            label_path: Path to corresponding label file
            
        Returns:
            Annotated image in RGB format, None if loading fails
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None
            
        height, width, _ = image.shape
        
        # Process annotations if label file exists
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    try:
                        # Extract normalized coordinates (skip class ID)
                        coords_normalized = [float(p) for p in parts[1:]]
                    except (ValueError, IndexError):
                        logger.warning(f"Error parsing line in {label_path}")
                        continue
                        
                    if len(coords_normalized) < 6:  # Need at least 3 points for polygon
                        continue
                        
                    # Convert normalized coordinates to pixel coordinates
                    polygon_pixels = []
                    for i in range(0, len(coords_normalized), 2):
                        x_normalized = coords_normalized[i]
                        y_normalized = coords_normalized[i + 1]
                        x_pixel = int(x_normalized * width)
                        y_pixel = int(y_normalized * height)
                        polygon_pixels.append([x_pixel, y_pixel])
                    
                    # Draw polygon
                    pts = np.array(polygon_pixels, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    
            except Exception as e:
                logger.error(f"Error processing annotations in {label_path}: {e}")
        
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @classmethod
    def display_random_images(
        cls, 
        image_dir: str, 
        label_dir: str, 
        num_images: int = 5,
        figsize: Tuple[int, int] = None
    ) -> None:
        """
        Display random images with their annotations.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing label files
            num_images: Number of images to display
            figsize: Figure size for matplotlib
        """
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            logger.error("One or both directories not found")
            return
            
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logger.error("No image files found in directory")
            return
            
        # Select random images
        random.shuffle(image_files)
        selected_files = image_files[:num_images]
        
        # Set up plot
        if figsize is None:
            figsize = (num_images * 4, 4)
            
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        if num_images == 1:
            axes = [axes]
            
        logger.info(f"Displaying {num_images} random images with annotations")
        
        for j, image_file in enumerate(selected_files):
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            # Load and annotate image
            annotated_image = cls.visualize_polygon(image_path, label_path)
            
            if annotated_image is not None:
                ax = axes[j]
                ax.imshow(annotated_image)
                ax.set_title(image_file, fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def convert_annotations_to_single_class(
        input_dir: str, 
        output_dir: str, 
        new_class_id: int = 0
    ) -> None:
        """
        Convert all annotations to use a single class ID.
        
        Args:
            input_dir: Directory containing original annotation files
            output_dir: Directory to save converted annotations
            new_class_id: New class ID to assign to all annotations
        """
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all label files
        label_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        if not label_files:
            logger.warning(f"No .txt label files found in {input_dir}")
            return
            
        logger.info(f"Found {len(label_files)} annotation files to process")
        
        processed_count = 0
        for i, label_file in enumerate(label_files):
            input_path = os.path.join(input_dir, label_file)
            output_path = os.path.join(output_dir, label_file)
            
            try:
                new_annotations = []
                with open(input_path, 'r') as infile:
                    for line in infile:
                        parts = line.strip().split()
                        if len(parts) >= 6:  # Valid annotation line
                            # Replace class ID with new one, keep coordinates
                            new_line = f"{new_class_id} " + " ".join(parts[1:])
                            new_annotations.append(new_line)
                
                # Write converted annotations
                with open(output_path, 'w') as outfile:
                    for annotation in new_annotations:
                        outfile.write(annotation + '\n')
                
                processed_count += 1
                
                # Log progress every 100 files
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(label_files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")
        
        logger.info(f"Annotation conversion complete! Processed {processed_count} files")
        logger.info(f"New labels saved to: {output_dir}")


class DatasetSplitter:
    """Handles dataset splitting for training, validation, and testing."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize dataset splitter.
        
        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def split_dataset(
        self,
        image_dir: str,
        label_dir: str,
        output_dir: str,
        dataset_name: str = "egyptian_id",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        class_names: List[str] = None
    ) -> str:
        """
        Split dataset into train/validation/test sets and create YOLO data.yaml.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            output_dir: Output directory for split dataset
            dataset_name: Name of the dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            class_names: List of class names
            
        Returns:
            Path to created data.yaml file
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        if class_names is None:
            class_names = ['egyptian_id']
        
        # Create output directory structure
        splits = ["train", "val", "test"]
        for split in splits:
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        
        # Get all image files
        all_images = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_images:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Shuffle for random split
        random.shuffle(all_images)
        
        # Calculate split sizes
        total_count = len(all_images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count
        
        # Split datasets
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]
        
        logger.info(f"Dataset split: Train={len(train_images)}, "
                   f"Val={len(val_images)}, Test={len(test_images)}")
        
        # Copy files to respective splits
        splits_data = [
            (train_images, "train"),
            (val_images, "val"), 
            (test_images, "test")
        ]
        
        for images_set, split_name in splits_data:
            logger.info(f"Copying {len(images_set)} files to '{split_name}' split...")
            
            for img_filename in images_set:
                # Source paths
                img_src = os.path.join(image_dir, img_filename)
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_src = os.path.join(label_dir, label_filename)
                
                # Destination paths
                img_dst = os.path.join(output_dir, "images", split_name, img_filename)
                label_dst = os.path.join(output_dir, "labels", split_name, label_filename)
                
                # Copy files
                try:
                    shutil.copy2(img_src, img_dst)
                    if os.path.exists(label_src):
                        shutil.copy2(label_src, label_dst)
                except Exception as e:
                    logger.error(f"Error copying {img_filename}: {e}")
        
        # Create data.yaml file
        yaml_path = os.path.join(output_dir, 'data.yaml')
        dataset_config = {
            'path': str(Path(output_dir).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, sort_keys=False, default_flow_style=False)
        
        logger.info(f"Dataset split complete! Created data.yaml at: {yaml_path}")
        return yaml_path