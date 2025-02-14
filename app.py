import gradio as gr
import torch
import PyPDF2
import os
import hashlib
from transformers import pipeline
from huggingface_hub import login
import config  # Import API keys from config.py

# Log in to Hugging Face using the token from config.py
login(token=config.HUGGINGFACE_TOKEN)

# Cache directory
CACHE_DIR = "cache_pdfs"
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the pipeline with GPU settings
if "pipe" not in globals():
    pipe = pipeline(
        "text-generation",
        model="ContactDoctor/Bio-Medical-Llama-3-8B",
        device="cuda",
        torch_dtype=torch.float16
    )

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
    messages = [
        {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
        {"role": "user", "content": text},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return outputs[0]["generated_text"]

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
    messages = [{"role": "user", "content": message}]
    response = pipe(messages)[0]["generated_text"]
    chat_history.append((message, response))
    return "", chat_history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# MedSense - Your Personalized Medical AI Intelligence")
    gr.Markdown("Upload PDF files for analysis or chat with your Medical AI assistant.")

    pdf_cache = gr.State({})
    with gr.Row():
        pdf_input = gr.Files(label="Upload PDF Files", file_types=[".pdf"])
        output = gr.Textbox(label="Model Analysis", lines=20)
    analyze_button = gr.Button("Analyze")
    analyze_button.click(fn=analyze_pdfs, inputs=[pdf_input, pdf_cache], outputs=[output, pdf_cache])

    gr.Markdown("---")

    chatbot = gr.Chatbot(label="Chat with AI")
    user_input = gr.Textbox(label="Enter your question", placeholder="Ask something...", lines=2)
    submit_button = gr.Button("Send")
    chat_history = gr.State([])

    submit_button.click(chat_with_model, inputs=[user_input, chat_history], outputs=[user_input, chatbot])

if __name__ == "__main__":
    demo.launch()
