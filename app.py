from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
import os

app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja modeli AI z pełną obsługą błędów
try:
    text_generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,
        torch_dtype="auto",
        framework="pt",  # Explicitly use PyTorch
        use_auth_token=False  # Disable if not using private models
    )
    # Immediately free up memory after initialization
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    text_generator = None

try:
    # Lżejszy model do podsumowań
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1,
        torch_dtype="auto"
    )
    logger.info("Model summarization załadowany pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modelu podsumowującego: {str(e)}")
    summarizer = None

import transformers
transformers.logging.set_verbosity_error()

# Configure Flask for production
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not summarizer:
        return render_template('error.html', message="Model podsumowujący nie jest dostępny")
    
    url = request.form.get('url', '').strip()
    if not url.startswith(('http://', 'https://')):
        return render_template('error.html', message="Nieprawidłowy URL")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')[:3] if p.get_text().strip()]
        text = ' '.join(paragraphs)
        
        if not text:
            return render_template('error.html', message="Nie znaleziono tekstu do analizy")
        
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return render_template('analysis.html', result=summary)
    
    except Exception as e:
        logger.error(f"Błąd analizy strony: {str(e)}")
        return render_template('error.html', message=f"Błąd analizy: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate():
    if not text_generator:
        return render_template('error.html', message="Model generujący nie jest dostępny")
    
    topic = request.form.get('topic', '').strip()
    if not topic:
        return render_template('error.html', message="Proszę podać temat")
    
    try:
        prompt = f"Napisz krótki post na Instagram (max 3 zdania) o temacie: {topic}"
        result = text_generator(prompt, max_length=200, num_return_sequences=1)
        post = result[0]['generated_text'].split('\n')[0]  # Pierwsza linia
        return render_template('generate.html', post=post)
    
    except Exception as e:
        logger.error(f"Błąd generowania posta: {str(e)}")
        return render_template('error.html', message=f"Błąd generowania: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render uses $PORT environment variable
    app.run(host='0.0.0.0', port=port, debug=False)