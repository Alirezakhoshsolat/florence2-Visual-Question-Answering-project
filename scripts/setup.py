#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Florence-2 VQA Project Setup Script

This script helps to set up the project environment, download the model,
and launch the Streamlit application for local development and testing.
Note: The application is already deployed and available at 
https://huggingface.co/spaces/parhamaki/data_mining_project

Usage:
    python scripts/setup.py [options]
    
Options:
    --no-venv       Skip virtual environment creation
    --no-download   Skip model download
    --install-only  Only install requirements without launching the app
"""

import os
import sys
import subprocess
import argparse
import platform

# Add the parent directory to the path so we can access other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    return True


def create_virtual_env(env_name="venv"):
    """Create a Python virtual environment."""
    print(f"Creating virtual environment '{env_name}'...")
    venv_path = os.path.join(ROOT_DIR, env_name)
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"Virtual environment '{env_name}' created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def install_requirements():
    """Install project requirements."""
    print("Installing project requirements...")
    
    # Path to requirements.txt
    req_path = os.path.join(ROOT_DIR, "requirements.txt")
    
    # Determine the pip command based on the platform and virtual environment
    if platform.system() == "Windows":
        pip_cmd = [os.path.join(ROOT_DIR, "venv", "Scripts", "pip")]
    else:
        pip_cmd = [os.path.join(ROOT_DIR, "venv", "bin", "pip")]
    
    try:
        # Upgrade pip
        subprocess.run([*pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([*pip_cmd, "install", "-r", req_path], check=True)
        
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def download_model():
    """Download the fine-tuned model."""
    print("Downloading the fine-tuned model...")
    
    # Path to download_model.py
    download_script = os.path.join(ROOT_DIR, "scripts", "download_model.py")
    
    # Determine the python command based on the platform and virtual environment
    if platform.system() == "Windows":
        python_cmd = [os.path.join(ROOT_DIR, "venv", "Scripts", "python")]
    else:
        python_cmd = [os.path.join(ROOT_DIR, "venv", "bin", "python")]
    
    try:
        subprocess.run([*python_cmd, download_script], check=True)
        print("Model downloaded successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        return False


def launch_app():
    """Launch the Streamlit app."""
    print("Launching the Streamlit application...")
    
    # Path to main.py
    app_path = os.path.join(ROOT_DIR, "app", "main.py")
    
    # Determine the streamlit command based on the platform and virtual environment
    if platform.system() == "Windows":
        streamlit_cmd = [os.path.join(ROOT_DIR, "venv", "Scripts", "streamlit")]
    else:
        streamlit_cmd = [os.path.join(ROOT_DIR, "venv", "bin", "streamlit")]
    
    try:
        subprocess.run([*streamlit_cmd, "run", app_path], check=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error launching application: {e}")
        return False
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        return True


def main():
    # Show information about the online application
    print("\n" + "="*80)
    print(" Florence-2 VQA Project - Local Setup ")
    print("="*80)
    print("\nNote: This application is already deployed online at:")
    print("https://huggingface.co/spaces/parhamaki/data_mining_project")
    print("\nThe following setup is only necessary if you want to run the application locally.")
    print("="*80 + "\n")

    parser = argparse.ArgumentParser(description="Set up Florence-2 VQA project")
    parser.add_argument(
        "--no-venv", 
        action="store_true", 
        help="Skip virtual environment creation"
    )
    parser.add_argument(
        "--no-download", 
        action="store_true", 
        help="Skip model download"
    )
    parser.add_argument(
        "--install-only", 
        action="store_true", 
        help="Only install requirements without launching the app"
    )
    
    args = parser.parse_args()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup process
    if not args.no_venv:
        if not create_virtual_env():
            sys.exit(1)
    
    if not install_requirements():
        sys.exit(1)
    
    if not args.no_download:
        if not download_model():
            sys.exit(1)
    
    if not args.install_only:
        launch_app()


if __name__ == "__main__":
    main()
