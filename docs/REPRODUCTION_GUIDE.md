# Reproduction Guide for Florence-2 VQA Project

This document provides detailed instructions for reproducing the Florence-2 VQA fine-tuning process and deploying the model using Streamlit.

## Environment Setup

The project was developed using the following environment:
- Python 3.8
- CUDA 11.7 (for GPU acceleration)
- PyTorch 2.0.1
- Transformers 4.31.0

## Dataset Preparation

1. **Download the VQA v2.0 dataset**
   - Visit https://visualqa.org/download.html
   - Download the Abstract Scenes subset:
     - Training questions
     - Training annotations
     - Validation questions
     - Validation annotations
     - Test questions
     - Training, validation, and test images

2. **Extract the dataset**
   ```bash
   unzip dataset.zip -d /content/dataset
   ```

3. **Preprocess the dataset**
   Run the preprocessing code in the `notebooks/florence_2_finetuned_vqa_dataset.ipynb` notebook. This will:
   - Combine questions with their corresponding answers
   - Format the data for the Florence-2 model
   - Create JSON files for training, validation, and testing

## Model Fine-tuning

Follow these steps to reproduce the fine-tuning process:

1. **Login to Hugging Face**
   ```python
   from huggingface_hub import login
   login()  # Enter your token when prompted
   ```

2. **Run the fine-tuning script**
   Execute the fine-tuning code in the notebook with the following parameters:
   - Learning rate: 1e-5
   - Batch size: 8 (adjust based on GPU memory)
   - Epochs: 3
   - Model: microsoft/Florence-2-base-ft

3. **Save checkpoints**
   The fine-tuning process will save checkpoints after each epoch. The final model will be saved to `/content/florence2_finetuned_model`.

## Model Evaluation

Evaluate the model using the validation set:
1. Load the fine-tuned model
2. Run inference on validation images
3. Compare answers with ground truth
4. Calculate accuracy, precision, and recall

## Deployment

1. **Prepare the model for deployment**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/florence2-vqa-project.git
   cd florence2-vqa-project
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Download the fine-tuned model
   python scripts/download_model.py
   ```

2. **Run the Streamlit application**
   ```bash
   # Run directly from the app directory
   streamlit run app/main.py
   
   # Or use the convenient wrapper script
   streamlit run app.py
   ```

3. **Deploy to Hugging Face Spaces**
   - Create a new Space on Hugging Face
   - Select Streamlit as the SDK
   - Upload the app.py, requirements.txt, and README.md files
   - Upload the fine-tuned model to the Space

## Troubleshooting

- **Memory issues during fine-tuning**: Reduce batch size or use gradient accumulation
- **CUDA out of memory**: Use mixed precision training with `torch.amp`
- **Slow inference**: Consider model quantization for deployment

## Citation

If you use this code or model, please cite:

```bibtex
@article{antol2015vqa,
  title={VQA: Visual Question Answering},
  author={Antol, Stanislaw and Agrawal, Aishwarya and Lu, Jiasen and Mitchell, Margaret and Batra, Dhruv and Zitnick, C Lawrence and Parikh, Devi},
  journal={Proceedings of the IEEE International Conference on Computer Vision},
  year={2015}
}
```
