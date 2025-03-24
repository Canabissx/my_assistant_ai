from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import logging

app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja modeli AI z pełną obsługą błędów
try:
    # Mniejszy model do generowania tekstu
    text_generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,  # Wymusza użycie CPU
        torch_dtype="auto",
        max_length=150  # Ograniczenie długości tekstu
    )
    logger.info("Model text-generation załadowany pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modelu generującego: {str(e)}")
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
    app.run(debug=False)  # Debug=False dla środowiska produkcyjnego