import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face API Key (from .env)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model Configuration
MODEL_ID = "ContactDoctor/Bio-Medical-Llama-3-8B"

# Generation Parameters
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
}

# Ensure API key is set
if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found. Make sure HUGGINGFACE_TOKEN is set in the .env file.")
