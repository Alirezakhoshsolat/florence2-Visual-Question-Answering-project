#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Florence-2 VQA Model Downloader

This script downloads the fine-tuned Florence-2 model from Hugging Face
and configures it for local use. Run this script before starting the 
Streamlit application.

Usage:
    python download_model.py
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

# Add the parent directory to the path so we can access other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define default model path relative to project root
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "florence2_finetuned_model")

def download_model(model_path=DEFAULT_MODEL_PATH, 
                  source="parhamaki/data_mining_project", 
                  verbose=True):
    """
    Download the fine-tuned model from Hugging Face.
    
    Args:
        model_path (str): Path to save the model
        source (str): Hugging Face repository ID
        verbose (bool): Print progress information
    
    Returns:
        bool: True if download was successful
    """
    try:
        print(f"Downloading Florence-2 model from {source}...")
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Download model files
        snapshot_download(
            repo_id=source,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            revision="main"
        )
        
        print(f"Model successfully downloaded to {model_path}")
        return True
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def verify_model(model_path="./florence2_finetuned_model"):
    """
    Verify the downloaded model by attempting to load it.
    
    Args:
        model_path (str): Path to the model
    
    Returns:
        bool: True if verification was successful
    """
    try:
        print("Verifying model files...")
        
        # Try to load model and processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        print("Model verification successful! The model is ready to use.")
        return True
    
    except Exception as e:
        print(f"Model verification failed: {e}")
        print("The downloaded model may be incomplete or corrupted.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Florence-2 fine-tuned model")
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the model"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="parhamaki/data_mining_project",
        help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--no-verify", 
        action="store_true",
        help="Skip model verification"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.path, args.source)
    
    if success and not args.no_verify:
        verify_model(args.path)

if __name__ == "__main__":
    main()
