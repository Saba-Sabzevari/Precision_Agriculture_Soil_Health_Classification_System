# Precision Agriculture Soil Health Classification System

## Project Overview
Hybrid expert system combining CNNs, metalearning, and rule-based reasoning for soil type classification from images.

**Course:** Pattern Recognition (M.Sc.) - Winter Semester 2025
**Instructor:** Prof. Raja Hashim Ali
**Group Members:**
- **Student 1 (Technical Lead):** Saba Sabzevari
- **Student 1 (Figures & Presentation):** Saba Sabzevari
- **Student 2 (Report & Storytelling):** Isha

## Dataset
- **Source:** [Kaggle Comprehensive Soil Classification Dataset](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets)
- **Classes:** 7 soil types (Alluvial, Arid, Black, Laterite, Mountain, Red, Yellow)
- **Total Images:** 1,186
- **Split:** 70% train, 15% val, 15% test (stratified)

## Model Architecture

### 1. CNN Feature Extraction (RQ1)
Three architectures compared:
- **ResNet50** (2048-dim features)
- **EfficientNet-B3** (1536-dim features) ✓ Best
- **Vision Transformer** (768-dim features)

### 2. Metalearner Ensemble (RQ2)
Stacking ensemble:
- Base learners: Random Forest, XGBoost, LightGBM, SVM
- Meta-learner: Logistic Regression

### 3. Rule-based Engine (RQ3)
Agricultural domain knowledge:
- Confidence thresholding
- Agreement validation
- Crop recommendations

## Results

| Configuration | Test Accuracy | Test F1-Score |
|--------------|---------------|---------------|
| ResNet50 | 79.21% | 70.81% |
| EfficientNet-B3 | 91.57% | 88.28% |
| Vision Transformer | 76.97% | 67.74% |
| Metalearner | 92.70% | 88.32% |
| **Hybrid System** | **92.70%** | **88.32%** |

## Explainability (RQ4)
- **GradCAM:** Visual attention maps for CNN predictions
- **SHAP:** Feature importance for metalearner

## Deployment (RQ5)
- **Inference Time:** 25.41 ms/batch (CPU)
- **Model Size:** 41.18 MB
- **Calibration:** ECE = 0.0752

## Reproduce Results

### Requirements
```bash
pip install torch torchvision scikit-learn xgboost lightgbm opencv-python shap
```

### Run
1. Mount Google Drive with dataset
2. Run all cells in `Soil_Classification.ipynb`
3. Checkpoints saved automatically - rerun uses cached models

### File Structure
```
Soil_Health_Classification_Phase2/
├── checkpoints/          # Trained models
├── figures/             # RQ1-RQ5 figures (PDF)
├── tables/              # RQ1-RQ5 tables (Excel)
└── dataset/             # Soil images
```

## References
- Dataset: [Kaggle Comprehensive Soil Classification](https://www.kaggle.com/datasets/ai4a-lab/comprehensive-soil-classification-datasets)
- EfficientNet: Tan & Le (2019)
- SHAP: Lundberg & Lee (2017)
- GradCAM: Selvaraju et al. (2017)
