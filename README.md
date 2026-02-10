#  🫁 Chest Cancer Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green.svg)](https://github.com/mayank2004201/Chest-Cancer-Classification)

An end-to-end deep learning project for automated classification of chest cancer using medical imaging. This project implements a complete MLOps pipeline with DVC for experiment tracking, modular code architecture, and a Flask-based web interface for predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Research & Development](#research--development)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Chest cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate detection is crucial for improving patient outcomes. This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to automatically classify chest CT scan images, assisting radiologists in making faster and more accurate diagnoses.

### Key Objectives

- ✅ Build a robust deep learning model for chest cancer classification
- ✅ Implement production-ready MLOps practices with DVC and modular code
- ✅ Create a user-friendly web interface for medical professionals
- ✅ Ensure reproducibility and version control for ML experiments
- ✅ Follow industry-standard project structure and best practices

---

## ✨ Features

### 🤖 Machine Learning
- **Transfer Learning** with pre-trained CNN architectures (VGG16/ResNet/EfficientNet)
- **Data Augmentation** for improved model generalization
- **Hyperparameter Tuning** for optimal performance
- **Model Evaluation** with comprehensive metrics (accuracy, precision, recall, F1-score, AUC-ROC)

### 🛠️ MLOps & Engineering
- **DVC Integration** for data versioning and pipeline management
- **Modular Architecture** with separation of concerns
- **Configuration Management** with YAML files
- **Logging & Monitoring** for tracking experiments
- **CI/CD Ready** with GitHub Actions workflow

### 🌐 Web Application
- **Flask-based Web Interface** for easy predictions
- **Image Upload Functionality** for chest scans
- **Real-time Predictions** with confidence scores
- **Responsive UI** with custom templates

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION                        │
│  (Download & Extract Medical Imaging Dataset)           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              DATA VALIDATION                            │
│  (Verify Data Integrity & Schema)                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          PREPARE BASE MODEL                             │
│  (Load Pre-trained CNN Architecture)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│               MODEL TRAINING                            │
│  (Fine-tune on Chest Cancer Dataset)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            MODEL EVALUATION                             │
│  (Test Performance & Generate Metrics)                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         WEB APPLICATION (Flask)                         │
│  (User Interface for Predictions)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core Technologies
| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Web Framework** | Flask |
| **Experiment Tracking** | DVC (Data Version Control) |
| **Configuration** | YAML |
| **Containerization** | Docker |
| **Version Control** | Git, GitHub |

### Key Libraries
```
tensorflow          # Deep learning framework
numpy              # Numerical computing
pandas             # Data manipulation
scikit-learn       # ML utilities
matplotlib         # Visualization
seaborn            # Statistical plots
flask              # Web application
dvc                # Data versioning
pyyaml             # Configuration management
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/mayank2004201/Chest-Cancer-Classification.git
cd Chest-Cancer-Classification
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Initialize DVC (Optional - for development)
```bash
dvc init
dvc pull  # Pull data if DVC remote is configured
```

---

## 🚀 Usage

### Training the Model

#### Option 1: Run Complete Pipeline
```bash
# Execute all stages: ingestion, validation, training, evaluation
python main.py
```

#### Option 2: Run Individual Stages with DVC
```bash
# Data ingestion
dvc repro data_ingestion

# Prepare base model
dvc repro prepare_base_model

# Model training
dvc repro training

# Model evaluation
dvc repro evaluation
```

### Making Predictions

#### Web Application
```bash
# Start the Flask server
python app.py
```
Then open your browser and navigate to:
```
http://localhost:8080
```

Upload a chest CT scan image and get instant predictions!

#### Programmatic Prediction
```python
from src.Chest_Cancer_Classification.pipeline.prediction import PredictionPipeline
from PIL import Image

# Load your image
image_path = "path/to/chest_scan.jpg"
image = Image.open(image_path)

# Create prediction pipeline
predictor = PredictionPipeline()

# Get prediction
result = predictor.predict(image)
print(f"Prediction: {result}")
```

---

## 📁 Project Structure

```
Chest-Cancer-Classification/
│
├── .dvc/                           # DVC configuration
├── .github/workflows/              # CI/CD pipelines
│   └── main.yml                    # GitHub Actions workflow
│
├── chest_cancer/                   # Artifacts directory
│   ├── data_ingestion/            # Downloaded datasets
│   ├── prepare_base_model/        # Pre-trained model files
│   ├── training/                  # Trained model checkpoints
│   └── evaluation/                # Evaluation results
│
├── config/                         # Configuration files
│   └── config.yaml                # Main configuration
│
├── model/                          # Final trained models
│   └── model.h5                   # Keras model file
│
├── research/                       # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb   # Data exploration
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/Chest_Cancer_Classification/
│   ├── components/                # Core components
│   │   ├── data_ingestion.py     # Data download & extraction
│   │   ├── prepare_base_model.py # Model architecture setup
│   │   ├── model_training.py     # Training logic
│   │   └── model_evaluation.py   # Evaluation metrics
│   │
│   ├── config/                    # Configuration management
│   │   └── configuration.py      # Config parser
│   │
│   ├── constants/                 # Constants
│   │   └── __init__.py
│   │
│   ├── entity/                    # Data classes
│   │   └── config_entity.py      # Configuration entities
│   │
│   ├── pipeline/                  # ML pipelines
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py         # Inference pipeline
│   │
│   └── utils/                     # Utility functions
│       └── common.py             # Helper functions
│
├── templates/                      # Flask HTML templates
│   ├── index.html                # Home page
│   └── results.html              # Prediction results
│
├── .gitignore                     # Git ignore file
├── app.py                         # Flask application
├── Dockerfile                     # Docker configuration
├── dvc.yaml                       # DVC pipeline definition
├── main.py                        # Main execution script
├── params.yaml                    # Model hyperparameters
├── requirements.txt               # Python dependencies
├── scores.json                    # Model evaluation scores
├── setup.py                       # Package setup
└── template.py                    # Project structure generator
```

---

## 🔄 ML Pipeline

### Pipeline Stages (Defined in `dvc.yaml`)

#### 1️⃣ Data Ingestion
**Purpose:** Download and extract chest cancer dataset  
**Outputs:** Raw medical images

**What it does:**
- Downloads dataset from configured source
- Validates data integrity
- Extracts and organizes images
- Splits into train/test sets

#### 2️⃣ Prepare Base Model
**Purpose:** Load and configure pre-trained CNN  
**Outputs:** Base model architecture

**What it does:**
- Loads pre-trained model (VGG16/ResNet/EfficientNet)
- Configures for transfer learning
- Freezes/unfreezes layers as per configuration
- Adds custom classification head

#### 3️⃣ Model Training
**Purpose:** Fine-tune model on chest cancer data  
**Outputs:** Trained model weights

**What it does:**
- Applies data augmentation
- Implements training callbacks (early stopping, model checkpointing)
- Logs training metrics
- Saves best model

#### 4️⃣ Model Evaluation
**Purpose:** Assess model performance  
**Outputs:** Evaluation metrics and visualizations

**What it does:**
- Generates predictions on test set
- Calculates metrics (accuracy, precision, recall, F1, AUC)
- Creates confusion matrix
- Saves results to `scores.json`

### Running the Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro 

# Visualize pipeline
dvc dag
```

---

## 📊 Model Performance

Final model performance metrics (stored in `scores.json`):

```json
{
  "loss": "0.7337068915367126"
  "accuracy": "0.6655518412590027",
}
```

> **Note:** Actual performance metrics depend on the dataset and training configuration. Update `scores.json` after training to reflect your results.

### Performance Visualization

The evaluation stage generates:
- ✅ Confusion Matrix
- ✅ ROC Curve
- ✅ Precision-Recall Curve
- ✅ Training History Plots

---

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
artifacts_root: chest_cancer

data_ingestion:
  root_dir: chest_cancer/data_ingestion
  source_URL: 
  local_data_file: chest_cancer/data_ingestion/data.zip
  unzip_dir: chest_cancer/data_ingestion

prepare_base_model:
  root_dir: chest_cancer/prepare_base_model
  base_model_path: chest_cancer/prepare_base_model/base_model.h5
  updated_base_model_path: chest_cancer/prepare_base_model/base_model_updated.h5

training:
  root_dir: chest_cancer/training
  trained_model_path: chest_cancer/training/model.h5

evaluation:
  root_dir: chest_cancer/evaluation
  mlflow_uri: ""
```

### Hyperparameters (`params.yaml`)

```yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
EPOCHS: 20
LEARNING_RATE: 0.001
CLASSES: 2

AUGMENTATION:
  rotation_range: 20
  horizontal_flip: true
  width_shift_range: 0.2
  height_shift_range: 0.2
  zoom_range: 0.2
```

---

## 🔬 Research & Development

The `research/` folder contains Jupyter notebooks documenting the experimental process:

1. **01_data_ingestion.ipynb**
   - Dataset exploration
   - Data distribution analysis
   - Sample visualization

2. **02_prepare_base_model.ipynb**
   - Model architecture experiments
   - Transfer learning strategies
   - Layer freezing analysis

3. **03_model_training.ipynb**
   - Training experiments
   - Hyperparameter tuning
   - Learning rate scheduling

4. **04_model_evaluation.ipynb**
   - Comprehensive performance analysis
   - Error analysis
   - Model interpretation

---

## 🔧 Development Workflow

This project follows a systematic development workflow:

### Standard Workflow Steps

1. **Update config.yaml** - Modify configuration parameters
2. **Update params.yaml** - Adjust model hyperparameters
3. **Update the entity** - Define data classes in `config_entity.py`
4. **Update configuration manager** - Parse configs in `configuration.py`
5. **Update components** - Implement core logic
6. **Update pipeline** - Create pipeline stages
7. **Update main.py** - Add execution logic
8. **Update dvc.yaml** - Define DVC pipeline stages

### Example: Adding a New Pipeline Stage

```python
# 1. Define config entity (entity/config_entity.py)
@dataclass
class NewStageConfig:
    root_dir: Path
    param1: str
    param2: int

# 2. Update configuration manager (config/configuration.py)
def get_new_stage_config(self) -> NewStageConfig:
    config = self.config.new_stage
    return NewStageConfig(
        root_dir=Path(config.root_dir),
        param1=config.param1,
        param2=config.param2
    )

# 3. Create component (components/new_stage.py)
class NewStageComponent:
    def __init__(self, config: NewStageConfig):
        self.config = config
    
    def execute(self):
        # Implementation
        pass

# 4. Create pipeline (pipeline/stage_05_new_stage.py)
class NewStagePipeline:
    def main(self):
        config = ConfigurationManager()
        stage_config = config.get_new_stage_config()
        component = NewStageComponent(config=stage_config)
        component.execute()

# 5. Add to main.py
if __name__ == '__main__':
    try:
        stage = NewStagePipeline()
        stage.main()
    except Exception as e:
        logger.exception(e)
        raise e

# 6. Define in dvc.yaml
stages:
  new_stage:
    cmd: python src/Chest_Cancer_Classification/pipeline/stage_05_new_stage.py
    deps:
      - src/Chest_Cancer_Classification/pipeline/stage_05_new_stage.py
    outs:
      - chest_cancer/new_stage/
```

---

## 🐳 Docker Support

### Build Docker Image
```bash
docker build -t chest-cancer-classifier .
```

### Run Container
```bash
docker run -p 8080:8080 chest-cancer-classifier
```

> **Note:** Cloud deployment has not been implemented. The Docker configuration is provided for local containerized execution.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Learning Resources

### Understanding the Code
- **MLOps Practices**: This project follows industry-standard MLOps practices
- **Modular Design**: Each component has a single responsibility
- **Configuration-Driven**: Easy to experiment without changing code
- **Reproducibility**: DVC ensures experiment reproducibility

### Key Concepts Demonstrated
- ✅ Transfer Learning for Medical Imaging
- ✅ End-to-End ML Pipeline Design
- ✅ Configuration Management
- ✅ Experiment Tracking with DVC
- ✅ Docker Containerization

---

## 🙏 Acknowledgments

- Medical imaging datasets from various open-source repositories
- Pre-trained models from TensorFlow/Keras
- MLOps best practices from [DVC.org](https://dvc.org/)
- Open-source deep learning community

---

## 📧 Contact

**Mayank**

- GitHub: [@mayank2004201](https://github.com/mayank2004201)
- Repository: [Chest-Cancer-Classification](https://github.com/mayank2004201/Chest-Cancer-Classification)

---

## 🔮 Future Enhancements

- Add MLflow for experiment tracking
- Implement model explainability (Grad-CAM, LIME)
- Multi-class classification for different cancer types
- REST API with FastAPI
- Model quantization for mobile deployment
- Integration with PACS systems
- Real-time monitoring dashboard
- A/B testing framework
- Cloud deployment (AWS/Azure/GCP)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for advancing medical AI

</div>
