# Lab-to-Field Crop Disease Detection

This project studies domain shift in crop disease classification between lab (PlantVillage) and real-world (PlantDoc) datasets.

## Key Features
- 17 aligned crop-disease classes
- EfficientNet-based classification pipeline
- Cross-domain evaluation
- Domain gap analysis (MMD, macro-F1)

## Results
- Source (PlantVillage): ~97% accuracy
- Target (PlantDoc): ~26% accuracy
- Significant domain gap observed

## Setup

1. Clone the repo:
   git clone https://github.com/your-username/repo-name.git

2. Install dependencies:
   pip install -r requirements.txt

3. Download datasets:
   - PlantVillage (Kaggle)
   - PlantDoc (Google Drive)

4. Run training:
   python train.py

## Notes
Datasets are not included due to size. Please download separately.