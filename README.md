# Hybrid CNN-Transformer Deepfake Detection

A computer vision project for detecting deepfake videos using a hybrid CNN-Transformer architecture. The system combines spatial feature extraction from video frames with temporal sequence modeling to identify inconsistencies commonly found in manipulated videos.

---

## Portfolio Highlights

- Built a deepfake detection pipeline for video classification
- Combined CNN-based spatial feature extraction with Transformer-based temporal analysis
- Supports EfficientNet-B0 and ResNet50 backbones
- Includes training, dataset preparation, and prediction scripts
- Uses attention mechanisms to focus on important spatial and temporal patterns
- Supports adversarial training for improved robustness

---

## Key Features

- **Hybrid Architecture**: CNN for frame-level visual features + Transformer for temporal reasoning
- **Backbone Flexibility**: EfficientNet-B0 and ResNet50 support
- **Temporal Modeling**: Multi-frame video analysis instead of single-image classification
- **Adversarial Training**: PGD-style training option to improve robustness
- **Attention Mechanisms**: Spatial and temporal attention for better feature learning
- **End-to-End Workflow**: Dataset preparation, model training, and inference scripts

---

## Tech Stack

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- EfficientNet / ResNet
- Transformer encoder architecture

---

## Project Structure

```text
deepfake-detection/
├── deepfake_detector.py          # Core hybrid CNN-Transformer model
├── train_deepfake_detector.py    # Training script
├── predict.py                    # Inference script for video prediction
├── prepare_dataset.py            # Dataset preparation utility
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignored local, dataset, and model files
└── README.md                     # Project documentation
```

---

## How It Works

1. A video is sampled into a fixed number of frames.
2. Each frame is passed through a CNN backbone to extract spatial features.
3. Spatial attention helps the model focus on important visual regions.
4. Frame-level features are passed into a Transformer to model temporal relationships.
5. The model predicts whether the video is real or fake.
6. During training, adversarial techniques can be used to improve robustness.

---

## Model Architecture

### 1. Spatial Feature Extractor

The CNN backbone extracts high-level features from each video frame.

Supported backbones:

- EfficientNet-B0
- ResNet50

### 2. Temporal Analyzer

The Transformer module analyzes the sequence of frame features and detects temporal inconsistencies across the video.

### 3. Classification Head

A fully connected classifier combines the learned spatial-temporal representation and outputs a real/fake prediction.

---

## Dataset Structure

The dataset should be organized as:

```text
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

Large datasets and trained model checkpoints should not be committed to GitHub. Keep them locally or host them externally.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mehdiiii2001/Deepfake-Detection.git
cd Deepfake-Detection
```

2. Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Prepare Dataset

```bash
python prepare_dataset.py
```

### Train Model

```bash
python train_deepfake_detector.py \
    --data_dir data \
    --batch_size 8 \
    --num_epochs 50 \
    --backbone efficientnet_b0
```

Common training options:

```text
--data_dir          Path to dataset
--batch_size        Batch size
--num_epochs        Number of training epochs
--lr                Learning rate
--num_frames        Number of frames sampled per video
--backbone          CNN backbone: efficientnet_b0 or resnet50
--checkpoint_dir    Directory for model checkpoints
--no_adversarial    Disable adversarial training
```

### Run Prediction

```bash
python predict.py \
    --input path/to/video.mp4 \
    --model_path checkpoints/best_model.pth
```

Prediction output includes:

- Real/Fake classification
- Confidence score
- Result visualization when supported

---

## Future Improvements

- Add evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC
- Add sample inference screenshots
- Add support for more deepfake datasets
- Add experiment tracking with TensorBoard or Weights & Biases
- Add pretrained model download instructions
- Add automated tests for preprocessing and model components
- Add Docker support for easier setup

---

## Why This Project Matters

Deepfake detection is an important computer vision problem because manipulated media can be used for misinformation, identity fraud, and social engineering. This project demonstrates an end-to-end approach to video-based deepfake classification using modern deep learning techniques.

---

## License

This project is licensed under the MIT License.
