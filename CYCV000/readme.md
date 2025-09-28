  # Egyptian ID Card Line Extraction System

This repository contains a complete computer vision pipeline for detecting Egyptian National ID cards in images, correcting their orientation, and segmenting them into individual lines of text.

The system is built using a YOLOv11 model for high-accuracy detection and a robust OpenCV pipeline for subsequent processing steps.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results and Demonstration](#results-and-demonstration)

## Project Overview

The objective of this assignment is to develop a system that extracts individual lines of text as cropped images from scanned or photographed ID documents. The pipeline is designed to be robust to variations in rotation, lighting, and background.

## Features

- **High-Accuracy Detection:** Utilizes a fine-tuned YOLOv11 model to reliably detect ID cards.
- **Automatic Orientation Correction:** Corrects for both fine rotational skew and coarse 90/180/270 degree rotations.
- **Background Removal:** Intelligently isolates the card from the background to prevent noise in the segmentation stage.
- **Line Segmentation:** Segments the cleaned card into individual, tightly-cropped text line images.

## System Architecture

1.  **ID Card Detection:** A YOLOv11 model trained on a large dataset of Egyptian IDs identifies and provides a bounding box for each card in the input image.
2.  **Alignment and Cropping:** The detected card is cropped and an OpenCV pipeline corrects its orientation using contour analysis and template matching.
3.  **Segmentation:** The aligned card is binarized using Otsu's thresholding, and contour grouping logic segments the card into text lines.

For a detailed breakdown of the model architecture, dataset choices, and performance, please refer to the `Technical_Report.md` file.

## Setup and Installation

This project is designed to be run in a Google Colab environment or a local environment with Python 3.8+ and the required dependencies.

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Install Dependencies:**

```
    pip install -r requirements.txt
 
```

Key dependencies include: ultralytics, opencv-python, numpy, and matplotlib.

**3. Download Model Weights:**
The fine-tuned YOLOv11 model weights (best_yolov11_id_model.pt) are included in this repository and in the root directory.

## How to Run

The easiest way to see the system in action and reproduce the results is by using the provided Jupyter Notebook.

### Using the Demo Notebook (demo.ipynb)

1. **Environment:** Open the demo.ipynb notebook in Google Colab or a local Jupyter environment.
2. **Upload Test Data:** Ensure the test_data/ directory is populated with your test images and that header_template.png is in the root directory.
3. **Run All Cells:** Execute all cells in the notebook from top to bottom.

The notebook will automatically initialize the pipeline, process each image in the test_data directory, and display the results inline. The visualization will show the original detected card followed by a gallery of its extracted text lines for a clear before-and-after comparison.