Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

link of the hosted files:
https://huggingface.co/spaces/parhamaki/data_mining_project/tree/main

# Florence-2 Visual Question Answering (VQA) System

![Florence-2 Model](https://img.shields.io/badge/model-Florence--2-blue)
![VQA Dataset](https://img.shields.io/badge/dataset-VQA%20v2.0-green)
![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)

This project demonstrates fine-tuning and deployment of Microsoft's Florence-2 multimodal model for Visual Question Answering (VQA) tasks. The system allows users to upload an image, ask questions about it, and receive AI-generated answers with confidence scores.

## 📌 Live Demo

Try the live application on Hugging Face Spaces:
[Florence-2 VQA Demo](https://huggingface.co/spaces/parhamaki/data_mining_project)

The application is fully deployed and ready to use at:
👉 **[https://huggingface.co/spaces/parhamaki/data_mining_project](https://huggingface.co/spaces/parhamaki/data_mining_project)**

No installation required - just visit the link above in your browser!

## 📋 Project Overview

This project involves:

1. **Fine-tuning** the Florence-2 model on the Abstract Scenes subset of the VQA v2.0 dataset
2. **Exploratory Data Analysis (EDA)** of the dataset
3. **Web Application** development using Streamlit for real-time inference
4. **Model Deployment** on Hugging Face Spaces

![Project Architecture](https://example.com/your-architecture-image.png)

## 🔧 Technical Components

- **Model**: Microsoft Florence-2 (multimodal vision-language model)
- **Fine-tuning**: Custom training loop with PyTorch
- **Dataset**: VQA v2.0 [Abstract Scenes](https://visualqa.org/)
- **Front-end**: Streamlit web application
- **Deployment**: Hugging Face Spaces

## 📊 Dataset

The Visual Question Answering (VQA) dataset v2.0 was used for training, which consists of:
- Open-ended questions about images
- Multiple possible answers for each question
- Abstract scenes subset (cartoon-like images)

Sample analysis of the dataset:
- Over 30,000 questions and images
- Question types include "what is", "how many", "is there", etc.
- Answer distribution analysis showed common responses

## 🚀 Model Fine-tuning

The Florence-2 model was fine-tuned using:
- Custom PyTorch training loop
- AdamW optimizer with learning rate of 1e-5
- 3 training epochs
- Beam search for inference
- Early stopping to prevent overfitting

The training workflow is documented in the `florence_2_finetuned_vqa_dataset.ipynb` notebook.

## 📱 Web Application

The Streamlit web application provides:
- Image upload functionality
- Interactive question input
- Model inference with confidence scores
- Session history tracking
- Results download capability

## 🔍 Results

The fine-tuned model shows strong performance on the VQA task:
- Improved accuracy compared to the base model
- Quick response time for real-time applications
- Ability to handle various question types and image scenarios

## 🛠️ Installation & Usage

> **Note**: The application is already deployed online at [Hugging Face Spaces](https://huggingface.co/spaces/parhamaki/data_mining_project). The following instructions are only necessary if you want to run the application locally for development or testing purposes.

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster inference)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/florence2-vqa-project.git
cd florence2-vqa-project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the fine-tuned model:
```bash
# Using the provided script
python scripts/download_model.py

# Or directly from Hugging Face
huggingface-cli download parhamaki/data_mining_project/florence2_finetuned_model --local-dir ./models/florence2_finetuned_model
```

### Running the Application

```bash
streamlit run app/main.py
```

Navigate to http://localhost:8501 in your web browser to use the application.

### Quick Setup

Alternatively, you can use the provided setup script which automates the installation process:

```bash
# Run the setup script
python setup.py

# Or with options
python setup.py --no-download  # Skip model download
python setup.py --install-only  # Don't launch the app
```

This script will create a virtual environment, install dependencies, download the model, and launch the application automatically.

## 📂 Repository Structure

The project follows a modular structure designed for clarity, maintenance, and extensibility:

```
florence2-vqa-project/
├── app/                           # Application code
│   ├── __init__.py                # Makes the directory a Python package
│   └── main.py                    # Main Streamlit application with VQA functionality
│
├── docs/                          # Documentation
│   ├── REPRODUCTION_GUIDE.md      # Step-by-step guide for reproducing the training process
│   └── Technical_report.pdf       # Technical documentation and analysis of the project
│
├── models/                        # Model directory (populated by download_model.py)
│   └── florence2_finetuned_model/ # Fine-tuned model files (download separately)
│
├── notebooks/                     # Jupyter notebooks
│   └── florence_2_finetuned_vqa_dataset.ipynb  # Training notebook with analysis and visualization
│
├── scripts/                       # Utility scripts
│   ├── __init__.py                # Makes the directory a Python package
│   ├── download_model.py          # Script to download the fine-tuned model from Hugging Face
│   └── setup.py                   # Setup script for environment and dependency management
│
├── .gitignore                     # Git ignore file for project-specific exclusions
├── .env.example                   # Example environment variables configuration
├── CITATION.bib                   # Academic citations and references
├── LICENSE                        # MIT License file
├── README.md                      # Main project documentation and user guide
├── requirements.txt               # Project dependencies for installation
└── setup.py                       # Root-level wrapper script for easy setup
```

### Directory Purposes:

- **app/**: Contains the web application code built with Streamlit that provides an interactive UI for the VQA system.
  
- **docs/**: Stores project documentation including reproduction guides and technical reports.
  
- **models/**: Directory where the fine-tuned Florence-2 model is stored after download. This directory is empty in the repository and populated during setup.
  
- **notebooks/**: Contains Jupyter notebooks used for model training, data preprocessing, and analysis.
  
- **scripts/**: Houses utility scripts for model download, environment setup, and other helper functions.

The root directory contains configuration files and entry point scripts that allow users to easily run the application without navigating through the directory structure.

## 🔗 References

- **Florence-2 Model**: [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft)
- **VQA Dataset**: [Visual Question Answering v2.0](https://visualqa.org/)
- **Paper**: Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Lawrence Zitnick, C., & Parikh, D. (2015). VQA: Visual question answering. *Proceedings of the IEEE International Conference on Computer Vision*, 2015, 2425-2433.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Microsoft Research](https://www.microsoft.com/en-us/research/) for the Florence-2 model
- [VQA Team](https://visualqa.org/people.html) for the dataset
- Hugging Face for hosting the model and demo

---

*This project was developed as part of a Data Mining and Machine Learning course.*
