# LoRA-Based Wildfire Detection and Risk Mapping Using Sentinel-2 Satellite Imagery and NASA FIRMS Data

## Overview

This project develops a geospatial artificial intelligence pipeline for wildfire detection and risk mapping using satellite imagery and deep learning.

The workflow combines:

- NASA FIRMS wildfire detections
- Sentinel-2 satellite imagery
- Google Earth Engine
- TensorFlow/Keras
- LoRA-inspired parameter-efficient fine-tuning

A pretrained EfficientNetV2B0 backbone is adapted using a custom low-rank adaptation (LoRA-style) layer to classify wildfire and non-wildfire satellite image tiles.

The project demonstrates how remote sensing and AI can be integrated for scalable environmental monitoring and urban resilience applications.

## Motivation

Wildfires are increasing in frequency and intensity due to climate change, drought conditions, and expanding urban-wildland interfaces.

Traditional monitoring systems rely heavily on thermal anomaly alerts and manual interpretation. This project explores whether satellite imagery combined with parameter-efficient deep learning can automatically identify wildfire-related spatial patterns and generate geospatial wildfire probability maps.

## Data Sources

### NASA FIRMS

Fire Information for Resource Management System (FIRMS)

Source:

https://firms.modaps.eosdis.nasa.gov/

Used for:

- Active wildfire detections
- Thermal anomaly identification
- Ground-truth label generation

Dataset:

- VIIRS SNPP Standard Processing (VIIRS_SNPP_SP)

### Sentinel-2

Source:

https://sentinel.esa.int/

Used for:

- High-resolution satellite imagery
- RGB image tile generation
- Wildfire visual pattern detection

Bands used:

- B4 (Red)
- B3 (Green)
- B2 (Blue)

### Google Earth Engine

Source:

https://earthengine.google.com/

Used for:

- Satellite image retrieval
- Cloud filtering
- Geospatial image processing

Dataset:

- COPERNICUS/S2_SR_HARMONIZED

## Study Area

California, USA

Bounding Box:

| Boundary | Value |
|-----------|---------|
| West | -124.5 |
| South | 32.4 |
| East | -114.0 |
| North | 42.1 |

Study Period:

July 2021 – October 2021

## Methodology

### 1. Wildfire Detection Retrieval

NASA FIRMS wildfire detections are downloaded using the FIRMS API.

High-confidence detections are retained.

### 2. Dataset Construction

Positive Samples:

- FIRMS wildfire detections

Negative Samples:

- Random locations spatially separated from known wildfire events

### 3. Satellite Image Retrieval

Sentinel-2 imagery is retrieved from Google Earth Engine.

For each sample:

- Cloud filtering applied
- RGB composite generated
- 224 × 224 image tile exported

### 4. Deep Learning Pipeline

Base Model:

- EfficientNetV2B0 (ImageNet pretrained)

Adaptation:

- Custom LoRA-inspired low-rank adaptation layer

Framework:

- TensorFlow / Keras

Advantages:

- Reduced trainable parameters
- Faster fine-tuning
- Lower memory requirements

## Model Architecture

Input Image
(224 × 224 RGB)

↓

EfficientNetV2B0 Backbone
(Frozen)

↓

Global Feature Embedding

↓

LoRA-Inspired Low-Rank Adaptation Layer

↓

Dense Classification Head

↓

Wildfire Probability

## Dataset Split

Stratified split:

| Dataset | Percentage |
|----------|------------|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

Class distributions were preserved using stratified sampling.

## Results

The trained model successfully learned wildfire-related visual patterns including:

- Smoke plumes
- Burn scars
- Vegetation damage
- Thermal anomaly regions

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- AUC
- Confusion Matrix

Outputs generated:

- Prediction CSV files
- Confusion matrix visualizations
- Probability distributions
- Interactive Folium wildfire maps

## Repository Structure

```text
sentinel2_firms_tf_lora/
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── wildfire_prediction_map.html
│   ├── fire_probability_distribution.png
│   └── test_metrics.json
│
├── model/
│   └── tf_lora_wildfire_model.keras
│
├── raw/
│   └── firms_fire_points.csv
│
├── tiles/
│   ├── fire/
│   └── normal/
│
├── train_metadata.csv
├── val_metadata.csv
├── test_metadata.csv
│
└── wildfire_detection.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/divyanatekar08/sentinel2-firms-lora-wildfire-detection.git

cd sentinel2-firms-lora-wildfire-detection
```

Install dependencies:

```bash
pip install tensorflow earthengine-api geemap folium pandas numpy scikit-learn pillow tqdm requests
```

## Required Credentials

### NASA FIRMS API Key

Obtain from:

https://firms.modaps.eosdis.nasa.gov/api/

Store as:

```text
FIRMS_MAP_KEY
```

### Google Earth Engine

Create a project and authenticate:

```bash
earthengine authenticate
```

Set:

```text
GEE_PROJECT
```

## Running the Project

Open:

```text
wildfire_detection.ipynb
```

Run all notebook cells sequentially.

The workflow will:

1. Download FIRMS detections
2. Generate wildfire and non-wildfire samples
3. Retrieve Sentinel-2 imagery
4. Train the LoRA-inspired model
5. Evaluate performance
6. Generate wildfire prediction maps

## Future Improvements

Potential extensions include:

- Multispectral Sentinel-2 bands
- Vision Transformers
- Temporal wildfire forecasting
- Weather and vegetation integration
- Semantic segmentation
- Explainable AI (Grad-CAM)
- Foundation models for Earth observation

## Author

Divya Natekar
