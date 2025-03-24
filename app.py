from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
import os
import torch  # Dodane na początku dla lepszego zarządzania pamięcią

app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optymalizacja zużycia pamięci
def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Inicjalizacja modeli AI z pełną obsługą błędów
try:
    text_generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,
        torch_dtype=torch.float16,  # Użyj 16-bitowej precyzji
        framework="pt",
        model_kwargs={
            "load_in_8bit": True,
            "low_cpu_mem_usage": True
        }
    )
    logger.info("Model generujący załadowany pomyślnie")
    cleanup_memory()
except Exception as e:
    logger.error(f"Błąd ładowania modelu generującego: {e}")
    text_generator = None

try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        device=-1,
        torch_dtype=torch.float16,  # Użyj 16-bitowej precyzji
        framework="pt",
        model_kwargs={
            "load_in_8bit": True,
            "low_cpu_mem_usage": True
        }
    )
    logger.info("Model podsumowujący załadowany pomyślnie")
    cleanup_memory()
except Exception as e:
    logger.error(f"Błąd ładowania modelu podsumowującego: {e}")
    summarizer = None

# Konfiguracja Flask dla produkcji
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit 1MB dla requestów

# Middleware do czyszczenia pamięci
@app.after_request
def clear_cache(response):
    cleanup_memory()
    return response

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
        
        prompt = f"Napisz krótki post (max 2 zdania) o: {topic}"
        result = text_generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7  # Kontrola kreatywności
        )
        post = result[0]['generated_text'].split('\n')[0]
        cleanup_memory()
        return render_template('generate.html', post=post)
    
    except Exception as e:
        logger.error(f"Błąd generowania: {str(e)}")
        return render_template('error.html', message=f"Błąd generowania: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        processes=1  # Ograniczenie do jednego procesu
    )