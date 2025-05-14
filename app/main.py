#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Florence-2 Visual Question Answering (VQA) Web Application

This application provides an interactive web interface for visual question answering
using a fine-tuned Florence-2 model. Users can upload images, ask questions about them,
and receive answers with confidence scores.

The model was fine-tuned on the VQA v2.0 dataset (Abstract Scenes).
GitHub: https://visualqa.org/

Author: Seyyedalireza Khosh solat
Date: May 2025
"""

import zipfile
import io
import base64
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import streamlit as st
import logging
import os
import sys

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MODEL_PATH = "../models/florence2_finetuned_model"
IMAGE_SIZE = (768, 768)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
VQA_TASK_PROMPT = "<VQA>"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Florence-2 VQA",
    page_icon="🖼️",
    layout="centered",
)

def load_model():
    """
    Load the fine-tuned Florence-2 model and processor.
    
    Returns:
        tuple: (model, processor) - The loaded model and processor
    """
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        ).to(device).eval()
        
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        logger.info("Model and processor loaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_resource
def get_model_and_processor():
    """Cached function to load the model and processor"""
    return load_model()

def process_image_and_question(image, question, num_beams=3, max_new_tokens=1024):
    """
    Process an image and question through the Florence-2 model.
    
    Args:
        image (PIL.Image): The input image
        question (str): The question about the image
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum new tokens to generate
        
    Returns:
        tuple: (answer, confidence_percentage) - The model's answer and confidence
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        full_prompt = f"{VQA_TASK_PROMPT} {question}"
        
        # Ensure the image is in RGB and properly resized
        image = image.convert("RGB").resize(IMAGE_SIZE)
        
        # Process inputs
        inputs = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        # Generate output with the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
        
        # Decode and extract answer
        generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        # Calculate confidence score
        confidence = torch.exp(outputs.sequences_scores).cpu().tolist()
        confidence_percentage = f"{confidence[0] * 100:.2f}%"
        
        # Post-process to get clean answer
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=VQA_TASK_PROMPT,
            image_size=(image.width, image.height)
        )
        answer = parsed_answer[VQA_TASK_PROMPT].replace("<pad>", "").strip()
        
        return answer, confidence_percentage
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        return f"Error: {str(e)}", "N/A"

def generate_history_download_link(history):
    """
    Generate a download link for the session history.
    
    Args:
        history (list): List of history entries with images and Q&A pairs
        
    Returns:
        str: HTML string containing download link
    """
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for idx, entry in enumerate(history):
                # Create text file with questions and answers
                question_answer_txt = "\n".join([
                    f"Q{qa_idx + 1}: {qa['question']}\nA{qa_idx + 1}: {qa['answer']} (Confidence: {qa['confidence']})"
                    for qa_idx, qa in enumerate(entry["qas"])
                ])
                zf.writestr(f"results_{idx + 1}.txt", question_answer_txt)
                
                # Add the image
                zf.writestr(f"image_{idx + 1}.png", entry["image_bytes"])
        
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.read()).decode()
        return f'<a href="data:application/zip;base64,{b64}" download="florence2_vqa_results.zip">Download Results</a>'
    
    except Exception as e:
        logger.error(f"Error generating download link: {e}")
        return "Failed to generate download link"

def save_to_history(image, question, answer, confidence):
    """
    Save the current Q&A pair to session history.
    
    Args:
        image (PIL.Image): The image being processed
        question (str): The question asked
        answer (str): The answer from the model
        confidence (str): The confidence percentage
    """
    # Convert image to bytes for storage
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Add to history or update existing entry
    if not st.session_state.history or st.session_state.history[-1]["image_bytes"] != img_bytes:
        st.session_state.history.append({
            "image_bytes": img_bytes,
            "qas": [{"question": question, "answer": answer, "confidence": confidence}]
        })
    else:
        st.session_state.history[-1]["qas"].append({"question": question, "answer": answer, "confidence": confidence})

def display_history():
    """Display the session history of questions and answers"""
    if not st.session_state.history:
        return
        
    st.write("### Question & Answer History")
    
    for idx, entry in enumerate(st.session_state.history):
        st.write(f"#### Results for Image {idx + 1}:")
        for qa in entry["qas"]:
            st.write(f"**Q:** {qa['question']} | **A:** {qa['answer']} (Confidence: {qa['confidence']})")
    
    # Download button for full history
    if st.button("Download Full History"):
        try:
            download_link = generate_history_download_link(st.session_state.history)
            st.markdown(download_link, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error during download: {e}")
            st.error(f"Error generating download: {e}")

def main():
    """Main application function"""
    # Display title and description
    st.title("🖼️ Florence-2 Visual Question Answering")
    st.write("Upload an image and ask a question about it. This application uses a fine-tuned "
             "Florence-2 model trained on the VQA v2.0 (Abstract Scenes) dataset.")
    
    # Initialize model and session state
    global model, processor
    model, processor = get_model_and_processor()
    
    # Initialize session state for history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Model Settings")
        num_beams = st.slider(
            "Number of Beams",
            min_value=1,
            max_value=10,
            value=3,
            help="Controls the diversity of generated answers. Higher values increase accuracy but slow down processing."
        )
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=50,
            max_value=1000,
            value=200,
            help="Sets the maximum number of tokens in the generated output. Useful for controlling the response length."
        )
        
        st.header("About")
        st.markdown("""
        This app demonstrates Visual Question Answering using the Florence-2 model 
        fine-tuned on the [VQA v2.0](https://visualqa.org/) dataset.
        
        - [GitHub Repository](https://github.com/yourusername/florence2-vqa-project)
        - [Hugging Face Space](https://huggingface.co/spaces/parhamaki/data_mining_project)
        """)
    
    # Main content area
    uploaded_image = st.file_uploader("Choose an image...", type=SUPPORTED_FORMATS)
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            question = st.text_input("What would you like to know about this image?")
            
            if st.button("Get Answer"):
                if not question.strip():
                    st.warning("Please enter a question about the image.")
                else:
                    with st.spinner("Processing your question..."):
                        answer, confidence = process_image_and_question(
                            image, question, num_beams, max_new_tokens
                        )
                    
                    # Display result with styling
                    st.markdown("### Answer:")
                    st.markdown(f"**{answer}**")
                    st.markdown(f"*Confidence: {confidence}*")
                    
                    # Save result to history
                    save_to_history(image, question, answer, confidence)
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            st.error(f"Error processing the image: {e}")
    
    # Display history
    display_history()


if __name__ == "__main__":
    main()
