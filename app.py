from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging
import os
import gc

app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Funkcja do czyszczenia pamięci
def cleanup_memory():
    gc.collect()

# Inicjalizacja modeli AI
try:
    text_generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",  # Lżejszy model
        device="cpu"
    )
    logger.info("✅ Model generujący załadowany pomyślnie")
except Exception as e:
    logger.error(f"❌ Błąd ładowania modelu generującego: {e}")
    text_generator = None

try:
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device="cpu"
    )
    logger.info("✅ Model podsumowujący załadowany pomyślnie")
except Exception as e:
    logger.error(f"❌ Błąd ładowania modelu podsumowującego: {e}")
    summarizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not summarizer:
        return render_template('error.html', message="❌ Model podsumowujący nie jest dostępny")
    
    url = request.form.get('url', '').strip()
    if not url.startswith(('http://', 'https://')):
        return render_template('error.html', message="❌ Nieprawidłowy URL")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # Uniknięcie blokad
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p', limit=3) if p.get_text().strip()]
        text = ' '.join(paragraphs) if paragraphs else "Nie znaleziono treści do analizy"

        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return render_template('analysis.html', result=summary)
    
    except requests.RequestException as e:
        logger.error(f"❌ Błąd pobierania strony: {e}")
        return render_template('error.html', message="❌ Nie udało się pobrać strony")

    except Exception as e:
        logger.error(f"❌ Błąd analizy: {e}")
        return render_template('error.html', message=f"❌ Wystąpił błąd analizy: {e}")

@app.route('/generate', methods=['POST'])
def generate():
    if not text_generator:
        return render_template('error.html', message="❌ Model generujący nie jest dostępny")
    
    topic = request.form.get('topic', '').strip()
    if not topic:
        return render_template('error.html', message="❌ Proszę podać temat")

    try:
        prompt = f"Napisz krótki post o: {topic}"
        result = text_generator(prompt, max_length=50, num_return_sequences=1, temperature=0.7)
        post = result[0]['generated_text'].split('\n')[0]
        return render_template('generate.html', post=post)
    
    except Exception as e:
        logger.error(f"❌ Błąd generowania: {e}")
        return render_template('error.html', message=f"❌ Wystąpił błąd generowania: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
