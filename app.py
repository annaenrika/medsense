import gradio as gr
import torch
import PyPDF2
import os
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import config  # Import the config file

# Log in to Hugging Face using the token from config.py
login(token=config.HUGGINGFACE_TOKEN)

# Cache directory
CACHE_DIR = "cache_pdfs"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load the model and tokenizer (cached in memory)
if "model" not in globals():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cpu",
    )

# Function to hash a file name (for caching)
def hash_filename(filename):
    return hashlib.md5(filename.encode()).hexdigest()

# Function to extract text from PDF files and cache results
def extract_text_from_pdf(pdf_file):
    hashed_name = hash_filename(pdf_file.name)
    cached_file = os.path.join(CACHE_DIR, f"{hashed_name}.txt")

    # If cached file exists, load it
    if os.path.exists(cached_file):
        with open(cached_file, "r", encoding="utf-8") as f:
            return f.read()

    # Otherwise, extract text and cache it
    with open(pdf_file.name, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join([page.extract_text() or "" for page in reader.pages])

    with open(cached_file, "w", encoding="utf-8") as f:
        f.write(text)

    return text

# Function to analyze text using the model
def analyze_text(text):
    messages = [
        {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
        {"role": "user", "content": text},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        eos_token_id=terminators,
        **config.GENERATION_CONFIG
    )
    return outputs[0]["generated_text"][len(prompt):]

# Function to handle PDF uploads and analysis
def analyze_pdfs(pdf_files, pdf_cache):
    combined_text = ""
    for pdf_file in pdf_files:
        hashed_name = hash_filename(pdf_file.name)
        if hashed_name in pdf_cache:
            extracted_text = pdf_cache[hashed_name]  # Load from cache
        else:
            extracted_text = extract_text_from_pdf(pdf_file)
            pdf_cache[hashed_name] = extracted_text  # Save to cache
        combined_text += extracted_text + "\n\n"
    return analyze_text(combined_text), pdf_cache

# Function to handle user chat input
def chat_with_model(message, chat_history):
    response = analyze_text(message)
    chat_history.append((message, response))
    return "", chat_history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# MedSense Your Personalized Medical AI Intelligence")
    gr.Markdown("Upload PDF files for analysis or chat with you Medical AI assistant.")

    # File upload section
    pdf_cache = gr.State({})  # Stores cached PDF content
    with gr.Row():
        pdf_input = gr.Files(label="Upload PDF Files", file_types=[".pdf"])
        output = gr.Textbox(label="Model Analysis", lines=20)
    analyze_button = gr.Button("Analyze")
    analyze_button.click(fn=analyze_pdfs, inputs=[pdf_input, pdf_cache], outputs=[output, pdf_cache])

    gr.Markdown("---")  # Separator

    # Chatbot section with persistent memory
    chatbot = gr.Chatbot(label="Chat with AI")
    user_input = gr.Textbox(label="Enter your question", placeholder="Ask something...", lines=2)
    submit_button = gr.Button("Send")
    chat_history = gr.State([])  # Stores chat memory

    submit_button.click(chat_with_model, inputs=[user_input, chat_history], outputs=[user_input, chatbot])

# Launch the Gradio app
demo.launch()
