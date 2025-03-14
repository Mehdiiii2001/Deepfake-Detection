# تشخیص ویدیوهای Deepfake با مدل ترکیبی CNN-Transformer
# Hybrid CNN-Transformer Deepfake Detection

[🇮🇷 Persian](#persian) | [🇺🇸 English](#english)

<a name="persian"></a>
## 🇮🇷 Persian

این پروژه یک سیستم هوشمند برای تشخیص ویدیوهای جعلی (Deepfake) با استفاده از یک مدل ترکیبی CNN-Transformer پیاده‌سازی کرده است.

### ویژگی‌های اصلی

- ترکیب CNN و Transformer برای تحلیل همزمان ویژگی‌های مکانی و زمانی
- پشتیبانی از شبکه‌های پایه مختلف (EfficientNet-B0 و ResNet50)
- آموزش متخاصم (Adversarial Training) برای مقابله با Deepfake‌های پیشرفته
- مکانیزم‌های توجه (Attention) مکانی و زمانی
- پردازش کارآمد فریم‌های ویدیویی

### پیش‌نیازها

```bash
# نصب وابستگی‌ها
pip install -r requirements.txt
```

### ساختار پروژه

```
deepfake-detection/
├── deepfake_detector.py     # پیاده‌سازی مدل اصلی
├── train_deepfake_detector.py   # اسکریپت آموزش
├── predict.py              # اسکریپت پیش‌بینی
├── prepare_dataset.py      # آماده‌سازی دیتاست
├── requirements.txt        # وابستگی‌های پروژه
└── README.md              # مستندات
```

### نحوه استفاده

#### ۱. آماده‌سازی محیط
```bash
# ایجاد محیط مجازی
python -m venv venv

# فعال‌سازی محیط مجازی در Windows
.\venv\Scripts\activate
# یا در Linux/Mac
source venv/bin/activate

# نصب وابستگی‌ها
pip install -r requirements.txt
```

#### ۲. آماده‌سازی دیتاست
```bash
python prepare_dataset.py
```

ساختار دیتاست باید به صورت زیر باشد:
```
data/
├── train/
│   ├── real/          # ویدیوهای واقعی برای آموزش
│   └── fake/          # ویدیوهای جعلی برای آموزش
└── val/
    ├── real/          # ویدیوهای واقعی برای اعتبارسنجی
    └── fake/          # ویدیوهای جعلی برای اعتبارسنجی
```

#### ۳. آموزش مدل
```bash
python train_deepfake_detector.py \
    --data_dir data \
    --batch_size 8 \
    --num_epochs 50 \
    --backbone efficientnet_b0
```

پارامترهای آموزش:
- `--data_dir`: مسیر دیتاست
- `--batch_size`: اندازه دسته (پیش‌فرض: ۸)
- `--num_epochs`: تعداد دوره‌های آموزش (پیش‌فرض: ۵۰)
- `--lr`: نرخ یادگیری (پیش‌فرض: ۰.۰۰۰۱)
- `--num_frames`: تعداد فریم‌های هر کلیپ (پیش‌فرض: ۱۶)
- `--backbone`: معماری CNN پایه (گزینه‌ها: efficientnet_b0، resnet50)
- `--checkpoint_dir`: مسیر ذخیره مدل (پیش‌فرض: checkpoints)
- `--no_adversarial`: غیرفعال کردن آموزش متخاصم

---

<a name="english"></a>
## 🇺🇸 English

This project implements an intelligent system for detecting deepfake videos using a hybrid CNN-Transformer model.

### Key Features

- Combined CNN and Transformer for simultaneous spatial-temporal analysis
- Support for multiple backbone networks (EfficientNet-B0 and ResNet50)
- Adversarial training to combat advanced deepfakes
- Spatial and temporal attention mechanisms
- Efficient video frame processing

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
deepfake-detection/
├── deepfake_detector.py     # Core model implementation
├── train_deepfake_detector.py   # Training script
├── predict.py              # Prediction script
├── prepare_dataset.py      # Dataset preparation
├── requirements.txt        # Project dependencies
└── README.md              # Documentation
```

### Usage

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment on Windows
.\venv\Scripts\activate
# or on Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Dataset Preparation
```bash
python prepare_dataset.py
```

The dataset should be structured as follows:
```
data/
├── train/
│   ├── real/          # Real videos for training
│   └── fake/          # Deepfake videos for training
└── val/
    ├── real/          # Real videos for validation
    └── fake/          # Deepfake videos for validation
```

#### 3. Model Training
```bash
python train_deepfake_detector.py \
    --data_dir data \
    --batch_size 8 \
    --num_epochs 50 \
    --backbone efficientnet_b0
```

Training parameters:
- `--data_dir`: Dataset path
- `--batch_size`: Batch size (default: 8)
- `--num_epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--num_frames`: Number of frames per clip (default: 16)
- `--backbone`: CNN backbone architecture (options: efficientnet_b0, resnet50)
- `--checkpoint_dir`: Model save path (default: checkpoints)
- `--no_adversarial`: Disable adversarial training

#### 4. Prediction
```bash
python predict.py \
    --input path/to/video.mp4 \
    --model_path checkpoints/best_model.pth
```

Prediction parameters:
- `--input`: Input video path
- `--model_path`: Trained model path
- `--num_frames`: Number of frames to analyze
- `--backbone`: CNN backbone architecture

### Model Architecture Details

#### 1. Spatial Feature Extractor (CNN)
- Uses EfficientNet-B0 or ResNet50
- Spatial attention mechanism
- High-level feature extraction from frames

#### 2. Temporal Analyzer (Transformer)
- Positional encoding for temporal information
- Multi-head self-attention mechanism
- Temporal inconsistency detection

#### 3. Fusion Mechanism
- Spatial and temporal feature fusion
- Fully connected layers with dropout
- Final sigmoid output

### Addressing Challenges

#### 1. Advanced Deepfakes
- PGD-based adversarial training
- Support for multiple CNN architectures
- Attention mechanisms for detail detection

#### 2. Dataset Bias
- Balanced data split (80/20)
- Data augmentation with transformations
- Independent validation

#### 3. Computational Efficiency
- Smart frame sampling
- Batch processing
- GPU memory optimization
- Learning rate scheduling

### Results and Output

For each input video, the system provides:
- Classification (REAL/FAKE)
- Confidence percentage
- Annotated result image

### Contributing

Your contributions to improve the project are valuable. Please:
1. Fork the project
2. Create a new branch
3. Make your changes
4. Submit a Pull Request

### License

This project is licensed under the MIT License.

### Support

For issues or suggestions, please create an Issue.