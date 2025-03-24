from flask import Flask, render_template, request
from transformers import pipeline  # Do generowania tekstu
import requests  # Do analizy stron i API
from bs4 import BeautifulSoup  # Do web scrapingu

app = Flask(__name__)

# Inicjalizacja modeli AI
text_generator = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    try:
        # Pobieranie i analiza strony
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')][:3])  # Pierwsze 3 akapity
        summary = summarizer(text, max_length=130)[0]['summary_text']
        return render_template('analysis.html', result=summary)
    except Exception as e:
        return f"Błąd: {str(e)}"

@app.route('/generate', methods=['POST'])
def generate():
    topic = request.form['topic']
    post = text_generator(f"Napisz post na Instagram o {topic}:", max_length=200)[0]['generated_text']
    return render_template('generate.html', post=post)

if __name__ == '__main__':
    app.run(debug=True)