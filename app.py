from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
import os
import torch
import gc

app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Czyści pamięć RAM i GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Wykrywanie GPU
USE_GPU = torch.cuda.is_available()
device = 0 if USE_GPU else -1

# Inicjalizacja modeli AI
try:
    text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=device)
    logger.info("Model generujący załadowany pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modelu generującego: {e}")
    text_generator = None

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
    logger.info("Model podsumowujący załadowany pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modelu podsumowującego: {e}")
    summarizer = None

@app.route('/')
def home():
    cleanup_memory()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not summarizer:
        return render_template('error.html', message="Model podsumowujący nie jest dostępny")
    
    try:
        url = request.form.get('url', '').strip()
        if not url.startswith(('http://', 'https://')):
            return render_template('error.html', message="Nieprawidłowy URL")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')[:3] if p.get_text().strip()]
        text = ' '.join(paragraphs) or "Nie znaleziono treści do analizy"
        
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        cleanup_memory()
        return render_template('analysis.html', result=summary)
    
    except Exception as e:
        logger.error(f"Błąd analizy: {str(e)}")
        return render_template('error.html', message=f"Błąd analizy: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate():
    if not text_generator:
        return render_template('error.html', message="Model generujący nie jest dostępny")
    
    try:
        topic = request.form.get('topic', '').strip()
        if not topic:
            return render_template('error.html', message="Proszę podać temat")
        
        prompt = f"Napisz krótki post o: {topic}"
        result = text_generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
        post = result[0]['generated_text'].split('\n')[0]
        cleanup_memory()
        return render_template('generate.html', post=post)
    
    except Exception as e:
        logger.error(f"Błąd generowania: {str(e)}")
        return render_template('error.html', message=f"Błąd generowania: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
