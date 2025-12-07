# Flood Risk Mapping from Satellite Images Using Deep Learning

**Project for Deep Learning Course (CSC3218)**  
**GROUP 3 PROJECT: Flood Risk Mapping from Satellite Images**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Models Implemented](#models-implemented)
8. [Results](#results)
9. [Key Features](#key-features)
10. [Technical Details](#technical-details)
11. [Future Work](#future-work)

---

## Project Overview

This project implements deep learning models for flood risk mapping from satellite images. The system classifies satellite images as **Flooded** or **Non-Flooded** using multiple deep learning architectures, enabling rapid flood detection and disaster response. This project was developed as part of the Deep Learning course (CSC3218) at Uganda Christian University, Faculty of Engineering, Design and Technology.

### Objectives

- Build deep learning models to classify satellite images into Flooded or Non-Flooded categories
- Compare baseline CNN and transfer learning models (ResNet50, InceptionV3) for performance
- Evaluate model performance using standard metrics (Accuracy, Precision, Recall, F1-score)
- Visualize results and provide insights for disaster management applications

### Justification

Automated flood detection from satellite imagery can:
- Enable rapid response during flood events
- Help in disaster risk assessment and planning
- Support humanitarian aid distribution
- Assist in insurance claim verification
- Provide valuable data for urban planning and infrastructure development

---

## Problem Statement

Flooding is a frequent natural disaster that impacts communities across Uganda and East Africa. Traditional flood monitoring methods are often time-consuming, expensive, and require expert analysis of satellite imagery. This project aims to develop an automated deep learning system that can classify satellite images as **Flooded** or **Non-Flooded**, enabling rapid flood detection and disaster response.

---

## Dataset

The project uses the **FloodNet Challenge - Track 1** dataset, which contains:

### Dataset Statistics
- **Training Images**: 398 total (51 Flooded, 347 Non-Flooded)
- **Validation Images**: 450 images
- **Test Images**: 448 images
- **Image Resolution**: Resized to 224×224 pixels
- **Image Format**: JPG files with corresponding PNG segmentation masks
- **Class Imbalance Ratio**: 1:6.8 (Flooded:Non-Flooded)

### Dataset Structure
```
FloodNet Challenge - Track 1/
├── Train/
│   ├── Labeled/
│   │   ├── Flooded/
│   │   │   ├── image/ (51 images)
│   │   │   └── mask/ (51 masks)
│   │   └── Non-Flooded/
│   │       ├── image/ (347 images)
│   │       └── mask/ (347 masks)
│   └── Unlabeled/
│       └── image/ (1047 images)
├── Validation/
│   └── image/ (450 images)
├── Test/
│   └── image/ (448 images)
└── class_mapping.csv
```

### Data Split
After preprocessing, the training data is split:
- **Training Set**: 318 samples (80%)
- **Validation Set**: 80 samples (20%)
- Stratified split to maintain class distribution

---

## Project Structure

```
.
├── project_notebook.ipynb          # Main Jupyter notebook with complete workflow
├── flood_prediction_ui.py          # Streamlit web application for predictions
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── PROJECT_REPORT.md               # Detailed project report
├── FloodNet Challenge - Track 1/   # Dataset folder
│   ├── Train/
│   ├── Validation/
│   ├── Test/
│   └── class_mapping.csv
├── best_baseline_model.keras       # Saved baseline CNN model
├── best_resnet50_model.keras      # Saved ResNet50 model
├── best_inceptionv3_model.keras   # Saved InceptionV3 model
├── baseline_history.json          # Training history for baseline model
├── resnet_history.json            # Training history for ResNet50
├── inception_history.json         # Training history for InceptionV3
├── evaluation_summary.json        # Evaluation metrics summary
└── learning_curves.png            # Visualization of training curves
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- (Optional) GPU with CUDA support for faster training

### Setup Steps

1. **Clone or download the project repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in the correct location:**
   - The dataset should be in the `FloodNet Challenge - Track 1` folder
   - Verify the folder structure matches the expected format
   - Download the dataset from: [FloodNet Challenge - Track 1](https://www.kaggle.com/datasets/hmendonca/floodnet)

4. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## Usage

### Running the Notebook

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook project_notebook.ipynb
   ```
   Or using JupyterLab:
   ```bash
   jupyter lab project_notebook.ipynb
   ```

2. **Execute cells sequentially:**
   - The notebook is organized into 9 main sections:
     1. Introduction
     2. Data Collection
     3. Data Preprocessing
     4. Exploratory Data Analysis (EDA)
     5. Model Development
     6. Model Evaluation
     7. Visualization & Discussion
     8. Conclusion & Recommendations
     9. References

3. **Training Models:**
   - **Baseline CNN**: Trained from scratch (~30 epochs)
   - **ResNet50**: Two-phase training (15 epochs frozen + 15 epochs fine-tuning)
   - **InceptionV3**: Two-phase training (15 epochs frozen + 15 epochs fine-tuning)

### Using Trained Models

After training, models are saved automatically. You can load and use them:

```python
import tensorflow as tf
from tensorflow import keras

# Load a saved model
model = keras.models.load_model('best_resnet50_model.keras')

# Make predictions
predictions = model.predict(image_array)
```

### Running the Streamlit Web Application

The project includes a user-friendly Streamlit web interface for making predictions:

1. **Launch the Streamlit app:**
   ```bash
   streamlit run flood_prediction_ui.py
   ```

2. **Use the interface:**
   - Upload a satellite/aerial image (JPG, PNG, BMP)
   - Click "Analyze Image" to get predictions
   - View ensemble and individual model predictions
   - See confidence scores and explanations
   - Analyze model agreement statistics

The web app automatically loads all trained models and provides an intuitive interface for flood detection from uploaded images.

---

## Models Implemented

### 1. Baseline CNN (From Scratch)
- **Architecture**: Custom convolutional neural network
- **Layers**:
  - 4 Convolutional blocks (32, 64, 128, 256 filters)
  - Batch Normalization and Dropout for regularization
  - Max Pooling layers
  - 2 Dense layers (512, 256 neurons)
  - Output layer with sigmoid activation
- **Total Parameters**: ~2.3M trainable parameters
- **Training**: Adam optimizer, learning rate 0.001

### 2. ResNet50 (Transfer Learning)
- **Base Model**: Pre-trained ResNet50 on ImageNet
- **Strategy**: Two-phase transfer learning
  - Phase 1: Frozen base model, train classifier head
  - Phase 2: Fine-tune top 20 layers of base model
- **Classifier**: Global Average Pooling + Dense layers (512, 256, 1)
- **Training**: Adam optimizer, learning rate 0.0001 (phase 1), 0.00001 (phase 2)

### 3. InceptionV3 (Transfer Learning)
- **Base Model**: Pre-trained InceptionV3 on ImageNet
- **Strategy**: Two-phase transfer learning
  - Phase 1: Frozen base model, train classifier head
  - Phase 2: Fine-tune top 30 layers of base model
- **Classifier**: Global Average Pooling + Dense layers (512, 256, 1)
- **Training**: Adam optimizer, learning rate 0.0001 (phase 1), 0.00001 (phase 2)

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Baseline CNN** | 87.50% | 0.8750 | 1.0000 | 0.9333 |
| **ResNet50** | 87.50% | 0.8750 | 1.0000 | 0.9333 |
| **InceptionV3** | 83.75% | 0.9831 | 0.8286 | 0.8992 |

### Key Findings

1. **Both Baseline CNN and ResNet50** achieved 87.5% accuracy on the validation set
2. **InceptionV3** showed better precision (98.31%) but lower recall (82.86%)
3. **Class Imbalance Impact**: Models show bias toward the majority class (Non-Flooded)
4. **Transfer Learning Benefits**: ResNet50 converged faster than baseline CNN
5. **Training Stability**: All models used early stopping and learning rate reduction

### Model Artifacts
- Training histories saved to JSON files
- Best model weights saved during training
- Learning curves visualization
- Confusion matrices for each model
- Sample prediction visualizations

---

## Key Features

### Data Preprocessing
- Image resizing to 224×224 pixels
- Normalization (pixel values scaled to [0, 1])
- Color space conversion (BGR to RGB)

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shifts: ±20%
- Shear transformation: ±20%
- Zoom: ±20%
- Horizontal and vertical flips
- Nearest neighbor fill mode

### Training Features
- **Early Stopping**: Prevents overfitting with patience=5
- **Learning Rate Reduction**: Adaptive LR reduction on plateau
- **Model Checkpointing**: Saves best model weights
- **Class Weighting**: Addresses class imbalance
- **Stratified Splitting**: Maintains class distribution

### Evaluation Metrics
- Accuracy
- Precision (per class and macro)
- Recall (per class and macro)
- F1-Score
- Confusion Matrix
- Classification Report

---

## Technical Details

### Environment
- **TensorFlow**: 2.20.0
- **Keras**: 3.12.0
- **Python**: 3.8+
- **NumPy**: 2.2.6
- **Image Processing**: OpenCV, PIL

### Hyperparameters
- **Image Size**: 224×224×3
- **Batch Size**: 32
- **Epochs**: 30 (Baseline), 15+15 (Transfer Learning)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Initial Learning Rate**: 0.001 (Baseline), 0.0001 (Transfer Learning)

### Computational Requirements
- **Training Time**: ~2-4 hours per model (depending on hardware)
- **Memory**: 8GB+ RAM recommended
- **GPU**: CUDA-enabled GPU recommended for faster training
- **Storage**: ~2GB for dataset + model files

---

## Future Work

1. **Expand Dataset**: Collect more diverse flood images from different regions and seasons
2. **Address Class Imbalance**: Implement SMOTE or advanced oversampling techniques
3. **Semantic Segmentation**: Extend to pixel-level segmentation for detailed flood mapping
4. **Semi-Supervised Learning**: Utilize unlabeled data (1047 images) to improve performance
5. **Real-time Deployment**: Develop web/mobile application for real-world deployment
6. **Multi-temporal Analysis**: Incorporate temporal information for flood progression tracking
7. **Ensemble Methods**: Combine predictions from multiple models
8. **Advanced Architectures**: Experiment with EfficientNet, Vision Transformers (ViT)

---

## Applications

### Disaster Response
- **Early Warning Systems**: Integrate models into automated monitoring systems
- **Rapid Assessment**: Use for quick damage assessment after flood events
- **Resource Allocation**: Guide humanitarian aid distribution based on flood extent

### Urban Planning
- **Risk Assessment**: Inform infrastructure development in flood-prone areas
- **Insurance Applications**: Assist in flood damage verification for insurance claims
- **Environmental Monitoring**: Long-term flood pattern analysis

---

## Project Requirements Completion

✓ **Dataset exploration and analysis (EDA)**  
✓ **Data preprocessing and augmentation**  
✓ **Baseline model (CNN from scratch)**  
✓ **Advanced models (Transfer Learning: ResNet50, InceptionV3)**  
✓ **Model evaluation with standard metrics**  
✓ **Visualizations (confusion matrix, learning curves, predictions)**  
✓ **Discussion and recommendations**  

---

## References

1. **FloodNet Dataset**: FloodNet Challenge - Track 1. Available at: https://www.kaggle.com/datasets/hmendonca/floodnet

2. **He, K., et al.** (2016). "Deep Residual Learning for Image Recognition." CVPR.

3. **Szegedy, C., et al.** (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR.

4. **Keras Documentation**: https://keras.io/

5. **TensorFlow Documentation**: https://www.tensorflow.org/

6. **ImageNet Dataset**: Deng, J., et al. (2009). "ImageNet: A large-scale hierarchical image database." CVPR.

---

## License

This project is for academic purposes as part of the Deep Learning course (CSC3218) at Uganda Christian University, Faculty of Engineering, Design and Technology, Advent 2025 Semester.

---

## Contact & Credits

**Institution**: Uganda Christian University  
**Faculty**: Engineering, Design and Technology  
**Course**: CSC3218 - Deep Learning  
**Semester**: Advent 2025  
**Project**: Flood Risk Mapping from Satellite Images  
**Group**: Group 3

---

## Interactive Web Application

This project includes a Streamlit-based web application (`flood_prediction_ui.py`) that provides:
- **Interactive Image Upload**: Easy drag-and-drop interface for satellite images
- **Real-time Predictions**: Instant flood detection analysis
- **Multi-model Ensemble**: Combines predictions from all trained models
- **Visual Analytics**: Charts and graphs showing prediction confidence
- **Detailed Explanations**: Context-aware descriptions of predictions
- **Model Comparison**: Side-by-side comparison of all model outputs

To run the web app:
```bash
pip install streamlit
streamlit run flood_prediction_ui.py
```

---

## Acknowledgments

- FloodNet Challenge organizers for providing the dataset
- TensorFlow and Keras development teams
- OpenCV and scientific Python community
- Streamlit team for the web framework





