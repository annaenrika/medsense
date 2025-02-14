import os
import warnings
import logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gradio as gr
import torch
import PyPDF2
import hashlib
from transformers import pipeline
from huggingface_hub import login
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device availability
device = "cpu"  # Default to CPU
print(f"Using device: {device}")

# Log in to Hugging Face
login(token=config.HUGGINGFACE_TOKEN)

# Cache directory
CACHE_DIR = "cache_pdfs"
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the pipeline
if "pipe" not in globals():
    try:
        pipe = pipeline(
            "text-generation",
            model=config.MODEL_ID,
            model_kwargs=config.MODEL_CONFIG
        )
        print("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def hash_filename(filename):
    return hashlib.md5(filename.encode()).hexdigest()

def extract_text_from_pdf(pdf_file):
    hashed_name = hash_filename(pdf_file.name)
    cached_file = os.path.join(CACHE_DIR, f"{hashed_name}.txt")

    if os.path.exists(cached_file):
        with open(cached_file, "r", encoding="utf-8") as f:
            return f.read()

    with open(pdf_file.name, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join([page.extract_text() or "" for page in reader.pages])

    with open(cached_file, "w", encoding="utf-8") as f:
        f.write(text)

    return text

def analyze_text(text):
    try:
        outputs = pipe(
            text,
            **config.GENERATION_CONFIG
        )
        return outputs[0]["generated_text"]
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        return f"Error generating response: {str(e)}"

def analyze_pdfs(pdf_files, pdf_cache):
    combined_text = ""
    for pdf_file in pdf_files:
        hashed_name = hash_filename(pdf_file.name)
        if hashed_name in pdf_cache:
            extracted_text = pdf_cache[hashed_name]
        else:
            extracted_text = extract_text_from_pdf(pdf_file)
            pdf_cache[hashed_name] = extracted_text
        combined_text += extracted_text + "\n\n"
    return analyze_text(combined_text), pdf_cache

def chat_with_model(message, chat_history):
    try:
        response = pipe(
            message,
            **config.GENERATION_CONFIG
        )[0]["generated_text"]
        chat_history.append((message, response))
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history.append((message, error_message))
    return "", chat_history

# Gradio interface
with gr.Blocks(
    title="MedSense AI",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# MedSense - Your Personalized Medical AI Intelligence")
    gr.Markdown("Upload PDF files for analysis or chat with your Medical AI assistant.")

    pdf_cache = gr.State({})
    with gr.Row():
        pdf_input = gr.Files(label="Upload PDF Files", file_types=[".pdf"])
        output = gr.Textbox(label="Model Analysis", lines=20)
    analyze_button = gr.Button("Analyze", variant="primary")
    analyze_button.click(fn=analyze_pdfs, inputs=[pdf_input, pdf_cache], outputs=[output, pdf_cache])

    gr.Markdown("---")

    chatbot = gr.Chatbot(label="Chat with AI", height=400)
    with gr.Row():
        user_input = gr.Textbox(
            label="Enter your question",
            placeholder="Ask something...",
            lines=2,
            scale=4
        )
        submit_button = gr.Button("Send", variant="primary", scale=1)
    chat_history = gr.State([])

    submit_button.click(
        chat_with_model,
        inputs=[user_input, chat_history],
        outputs=[user_input, chatbot]
    )

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio server...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True,
            auth=None,
            ssl_verify=False,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
