import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Data class for bounding box coordinates with confidence score."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float


@dataclass
class ProcessingConfig:
    """Configuration parameters for the ID processing pipeline."""
    # Detection parameters
    detection_confidence: float = 0.25
    crop_margin: int = 30
    
    # Line segmentation parameters
    roi_start_ratio: float = 0.35  
    contour_area_threshold: int = 20
    line_grouping_threshold: int = 20
    line_margin: int = 5
    
    # Adaptive threshold parameters
    adaptive_block_size: int = 25
    adaptive_c_value: int = 10
    
    # Line filtering parameters
    min_line_height: int = 10
    max_line_height: int = 150
    
    # Template matching parameters
    rotation_angles: List[int] = None
    
    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [0, 90, 180, 270]


class IDCardDetector:
    """Handles ID card detection using YOLO model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the ID card detector.
        
        Args:
            model_path: Path to the trained YOLO model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            self.model = YOLO(model_path)
            logger.info(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_cards(self, image_path: str, confidence: float = 0.5) -> Optional[List[BoundingBox]]:
        """
        Detect ID cards in the given image.
        
        Args:
            image_path: Path to the input image
            confidence: Minimum confidence threshold for detection
            
        Returns:
            List of BoundingBox objects for detected cards, None if no cards found
        """
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return None
            
        try:
            results = self.model.predict(source=image_path, conf=confidence, save=False)
            
            if not results or len(results[0].boxes) == 0:
                logger.warning("No ID cards detected in the image")
                return None
                
            bounding_boxes = []
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                bbox = BoundingBox(
                    x_min=int(coords[0]),
                    y_min=int(coords[1]),
                    x_max=int(coords[2]),
                    y_max=int(coords[3]),
                    confidence=float(box.conf[0])
                )
                bounding_boxes.append(bbox)
                
            logger.info(f"Detected {len(bounding_boxes)} ID card(s)")
            return bounding_boxes
            
        except Exception as e:
            logger.error(f"Error during card detection: {e}")
            return None


class ImageProcessor:
    """Handles image rotation and orientation correction."""
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by the specified angle with proper boundary handling.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees (positive = clockwise)
            
        Returns:
            Rotated image array
        """
        if angle == 0:
            return image.copy()
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    @staticmethod
    def detect_skew_angle(image: np.ndarray) -> float:
        """
        Detect fine skew angle using contour analysis.
        
        Args:
            image: Input image array (RGB)
            
        Returns:
            Detected skew angle in degrees
        """
        try:
            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found for skew detection")
                return 0.0
                
            # Get largest contour (presumably the ID card boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Normalize angle to [-45, 45] range
            width, height = rect[1]
            if width < height:
                angle += 90
                
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
                
            logger.debug(f"Detected skew angle: {angle:.2f}°")
            return angle
            
        except Exception as e:
            logger.error(f"Error detecting skew angle: {e}")
            return 0.0


class OrientationCorrector:
    """Handles ID card orientation correction using template matching."""
    
    def __init__(self, template_path: str):
        """
        Initialize orientation corrector with template image.
        
        Args:
            template_path: Path to the template image for orientation detection

        """
        if not Path(template_path).exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Could not load template image from {template_path}")
            
        logger.info(f"Loaded orientation template: {template_path}")
    
    def correct_orientation(
        self, 
        cropped_image: np.ndarray, 
        config: ProcessingConfig
    ) -> np.ndarray:
        """
        Correct the orientation of a cropped ID card image.
        
        Args:
            cropped_image: Cropped ID card image
            config: Processing configuration
            
        Returns:
            Orientation-corrected image
        """
        # Step 1: Detect fine skew angle
        fine_skew = ImageProcessor.detect_skew_angle(cropped_image)
        logger.debug(f"Fine skew detected: {fine_skew:.2f}°")
        
        # Step 2: Test different orientation hypotheses
        best_angle = fine_skew
        highest_score = -1.0
        
        logger.debug("Testing orientation hypotheses...")
        for coarse_angle in config.rotation_angles:
            test_angle = fine_skew + coarse_angle
            
            # Rotate image for testing
            rotated = ImageProcessor.rotate_image(cropped_image, test_angle)
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
            
            # Skip if image is too small for template matching
            if (rotated_gray.shape[0] < self.template.shape[0] or 
                rotated_gray.shape[1] < self.template.shape[1]):
                continue
                
            # Perform template matching
            result = cv2.matchTemplate(rotated_gray, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            logger.debug(f"Angle {test_angle:.1f}°: score = {max_val:.4f}")
            
            if max_val > highest_score:
                highest_score = max_val
                best_angle = test_angle
        
        logger.info(f"Best orientation: {best_angle:.2f}° (score: {highest_score:.4f})")
        
        # Apply final rotation
        return ImageProcessor.rotate_image(cropped_image, best_angle)


class LineSegmenter:
    """Handles line segmentation from ID card images."""
    
    @staticmethod
    def segment_lines(image: np.ndarray, config: ProcessingConfig) -> List[np.ndarray]:
        """
        Segment ID card image into individual text lines using contour grouping.
        
        Args:
            image: Input ID card image
            config: Processing configuration
            
        Returns:
            List of segmented line images
        """
        height, width = image.shape[:2]
        
        # Define ROI (skip the leftmost part containing photo/logo)
        roi_start_x = int(width * config.roi_start_ratio)
        roi_image = image[:, roi_start_x:]
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            config.adaptive_block_size, config.adaptive_c_value
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get bounding boxes
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > config.contour_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust x coordinate to account for ROI offset
                bounding_boxes.append((x + roi_start_x, y, w, h))
        
        if not bounding_boxes:
            logger.warning("No valid contours found for line segmentation")
            return []
        
        # Group bounding boxes into lines based on vertical proximity
        bounding_boxes.sort(key=lambda box: box[1])  # Sort by y-coordinate
        
        line_groups = []
        current_group = []
        
        for box in bounding_boxes:
            if not current_group:
                current_group.append(box)
            else:
                last_box = current_group[-1]
                # Check if boxes are on the same line (similar y-coordinates)
                if abs(box[1] - last_box[1]) < config.line_grouping_threshold:
                    current_group.append(box)
                else:
                    line_groups.append(current_group)
                    current_group = [box]
        
        if current_group:  # Don't forget the last group
            line_groups.append(current_group)
        
        # Extract line images
        segmented_lines = []
        for group in line_groups:
            if not group:
                continue
                
            # Calculate bounding rectangle for the entire line
            x_min = min(box[0] for box in group)
            y_min = min(box[1] for box in group)
            x_max = max(box[0] + box[2] for box in group)
            y_max = max(box[1] + box[3] for box in group)
            
            # Add margin and ensure bounds are valid
            y_min = max(0, y_min - config.line_margin)
            y_max = min(height, y_max + config.line_margin)
            
            # Extract line image (full width)
            line_image = image[y_min:y_max, :]
            segmented_lines.append(line_image)
        
        logger.info(f"Segmented {len(segmented_lines)} lines")
        return segmented_lines
    
    @staticmethod
    def filter_lines_by_height(
        lines: List[np.ndarray], 
        min_height: int, 
        max_height: int
    ) -> List[np.ndarray]:
        """
        Filter segmented lines based on height criteria.
        
        Args:
            lines: List of line images
            min_height: Minimum acceptable line height
            max_height: Maximum acceptable line height
            
        Returns:
            Filtered list of line images
        """
        filtered_lines = []
        removed_count = 0
        
        for line in lines:
            line_height = line.shape[0]
            if min_height <= line_height <= max_height:
                filtered_lines.append(line)
            else:
                removed_count += 1
                logger.debug(f"Removed line with height {line_height}px")
        
        logger.info(f"Kept {len(filtered_lines)} lines, removed {removed_count} lines")
        return filtered_lines


class IDLineExtractionPipeline:
    """Main pipeline for Egyptian ID card line extraction."""
    
    def __init__(self, model_path: str, template_path: str, config: ProcessingConfig = None):
        """
        Initialize the extraction pipeline.
        
        Args:
            model_path: Path to YOLO model file
            template_path: Path to template image for orientation correction
            config: Processing configuration (uses default if None)
        """
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.detector = IDCardDetector(model_path)
        self.corrector = OrientationCorrector(template_path)
        self.segmenter = LineSegmenter()
        
        logger.info("ID line extraction pipeline initialized successfully")
    
    def process_image(self, image_path: str) -> Optional[List[Tuple[int, List[np.ndarray]]]]:
        """
        Process a single image and extract text lines from all detected ID cards.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of tuples containing (card_index, list_of_line_images) for each card
        """
        if not Path(image_path).exists():
            logger.error(f"Input image not found: {image_path}")
            return None
        
        try:
            # Step 1: Detect ID cards
            logger.info("Step 1: Detecting ID cards...")
            bounding_boxes = self.detector.detect_cards(
                image_path, self.config.detection_confidence
            )
            
            if not bounding_boxes:
                return None
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Step 2: Process each detected card
            results = []
            for card_idx, bbox in enumerate(bounding_boxes):
                logger.info(f"Processing card {card_idx + 1}/{len(bounding_boxes)}")
                
                # Crop ID card with margin
                x_min = max(0, bbox.x_min - self.config.crop_margin)
                y_min = max(0, bbox.y_min - self.config.crop_margin)
                x_max = min(image_rgb.shape[1], bbox.x_max + self.config.crop_margin)
                y_max = min(image_rgb.shape[0], bbox.y_max + self.config.crop_margin)
                
                cropped_id = image_rgb[y_min:y_max, x_min:x_max].copy()
                
                # Step 3: Correct orientation
                logger.info("Step 2: Correcting orientation...")
                oriented_id = self.corrector.correct_orientation(cropped_id, self.config)
                
                # Step 4: Segment lines
                logger.info("Step 3: Segmenting text lines...")
                raw_lines = self.segmenter.segment_lines(oriented_id, self.config)
                
                # Step 5: Filter lines
                final_lines = self.segmenter.filter_lines_by_height(
                    raw_lines, self.config.min_line_height, self.config.max_line_height
                )
                
                results.append((card_idx, final_lines))
                logger.info(f"Card {card_idx + 1}: extracted {len(final_lines)} text lines")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def visualize_results(self, results: List[Tuple[int, List[np.ndarray]]]):
        """
        Display the extraction results using matplotlib.
        
        Args:
            results: Results from process_image method
        """
        for card_idx, line_images in results:
            if not line_images:
                logger.warning(f"No lines found for card {card_idx + 1}")
                continue
            
            # Display each line
            for line_idx, line_img in enumerate(line_images):
                plt.figure(figsize=(12, 3))
                plt.imshow(line_img)
                plt.title(f"Card {card_idx + 1} - Line {line_idx + 1}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()


def main():
    """
    Main execution function demonstrating the ID line extraction pipeline.
    """
    # Configuration
    MODEL_PATH = 'model path'
    TEMPLATE_PATH = 'header_template.png'
    TEST_IMAGE_PATH = 'image'
    
    # Verify required files exist
    required_files = [MODEL_PATH, TEMPLATE_PATH, TEST_IMAGE_PATH]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return
    
    try:
        # Create custom configuration if needed
        config = ProcessingConfig(
            detection_confidence=0.5,
            crop_margin=30,
            min_line_height=30,
            max_line_height=140
        )
        
        # Initialize pipeline
        logger.info("Initializing ID line extraction pipeline...")
        pipeline = IDLineExtractionPipeline(MODEL_PATH, TEMPLATE_PATH, config)
        
        # Process image
        logger.info(f"Processing image: {TEST_IMAGE_PATH}")
        results = pipeline.process_image(TEST_IMAGE_PATH)
        
        if results:
            logger.info("Processing completed successfully!")
            
            # Display results
            pipeline.visualize_results(results)
            
            # Print summary
            total_lines = sum(len(lines) for _, lines in results)
            logger.info(f"Summary: Extracted {total_lines} text lines from {len(results)} ID card(s)")
        else:
            logger.warning("No results obtained from processing")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


if __name__ == '__main__':
    main()