from flask import Flask, render_template, request, jsonify
from models import NLP

chat = NLP()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    text = data.get('text')

    text = chat.answer(text)
    response = text

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)