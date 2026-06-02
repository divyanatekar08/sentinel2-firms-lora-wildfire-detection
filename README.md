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

Wildfires continue to increase in frequency and severity due to climate change, drought conditions, and expanding urban-wildland interfaces.

Traditional monitoring systems rely heavily on thermal anomaly alerts and manual interpretation. This project explores whether satellite imagery combined with parameter-efficient deep learning can automatically identify wildfire-related spatial patterns and generate geospatial wildfire probability maps.

## Data Sources

### NASA FIRMS

Fire Information for Resource Management System (FIRMS)

Used for:

- Active wildfire detections
- Thermal anomaly identification
- Ground-truth label generation

Dataset:

- VIIRS SNPP Standard Processing (VIIRS_SNPP_SP)

Source:

https://firms.modaps.eosdis.nasa.gov/

### Sentinel-2

Used for:

- Satellite image retrieval
- Wildfire image tile generation
- Environmental feature extraction

Bands Used:

- B4 (Red)
- B3 (Green)
- B2 (Blue)

Source:

https://sentinel.esa.int/

### Google Earth Engine

Used for:

- Satellite image acquisition
- Cloud filtering
- Geospatial processing

Dataset:

- COPERNICUS/S2_SR_HARMONIZED

Source:

https://earthengine.google.com/

## Study Area

California, USA

Bounding Coordinates:

| Boundary | Value |
|-----------|---------|
| West | -124.5 |
| South | 32.4 |
| East | -114.0 |
| North | 42.1 |

Study Period:

July 2021 – October 2021

## Methodology

### Step 1

Download wildfire detections from NASA FIRMS.

### Step 2

Generate wildfire and non-wildfire locations.

### Step 3

Retrieve Sentinel-2 satellite imagery using Google Earth Engine.

### Step 4

Create a labeled satellite image dataset.

### Step 5

Train a LoRA-inspired EfficientNetV2B0 classifier.

### Step 6

Evaluate model performance.

### Step 7

Generate geospatial wildfire prediction maps.

## Model Architecture

Input Image (224×224 RGB)

↓

EfficientNetV2B0 Backbone

↓

Global Feature Extraction

↓

LoRA-Inspired Low-Rank Adaptation Layer

↓

Dense Classification Head

↓

Wildfire Probability

## Dataset Split

The dataset was split using stratified sampling.

| Dataset | Percentage |
|----------|------------|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

This preserves wildfire and non-wildfire class distributions across all subsets.

## Results

The model learned wildfire-related visual patterns including:

- Smoke plumes
- Burn scars
- Vegetation damage
- Thermal anomaly regions

Generated outputs include:

- Confusion Matrix
- Probability Distribution Plots
- Test Metrics
- Interactive Wildfire Prediction Maps

## Repository Structure

```text
sentinel2_firms_tf_lora/

├── SENTINEL2_FIRMS_TF_LORA_PROGRAM.ipynb
│
├── model/
│   └── trained TensorFlow model
│
├── outputs/
│   ├── confusion matrix
│   ├── prediction results
│   ├── wildfire maps
│   └── evaluation metrics
│
├── raw/
│   └── downloaded wildfire detections
│
├── tiles/
│   ├── fire
│   └── normal
│
├── samples.csv
├── tile_metadata.csv
├── train_metadata.csv
├── val_metadata.csv
└── test_metadata.csv
```

## Technologies Used

- Python
- TensorFlow / Keras
- Google Earth Engine
- NASA FIRMS API
- Pandas
- NumPy
- Scikit-Learn
- Folium
- Pillow
- Requests

## Running the Project

1. Configure Google Earth Engine authentication.
2. Add your FIRMS API key.
3. Open the notebook:

```bash
jupyter notebook SENTINEL2_FIRMS_TF_LORA_PROGRAM.ipynb
```

4. Run notebook cells sequentially.

The notebook will:

- Download wildfire detections
- Retrieve Sentinel-2 imagery
- Generate training data
- Train the model
- Evaluate results
- Create wildfire prediction maps

## Future Improvements

Potential extensions include:

- Multispectral Sentinel-2 bands
- Vision Transformers
- Temporal wildfire forecasting
- Weather data integration
- Vegetation index features
- Explainable AI (Grad-CAM)
- Earth observation foundation models

## Author

Divya Natekar

M.S. Urban Data Science

New York University (NYU)
