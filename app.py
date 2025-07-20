from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from functools import wraps
import time

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'APIKEY')
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def rate_limit(max_per_minute=30):
    def decorator(f):
        last_called = [0.0]
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = (60.0 / max_per_minute) - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            last_called[0] = time.time()
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@rate_limit(max_per_minute=30)
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No se recibió ningún mensaje'
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'El mensaje no puede estar vacío'
            }), 400
        
        if len(user_message) > 2000:
            return jsonify({
                'success': False,
                'error': 'El mensaje es demasiado largo (máximo 2000 caracteres)'
            }), 400
        
        response = model.generate_content(user_message)
        
        if not response.text:
            return jsonify({
                'success': False,
                'error': 'No se pudo generar una respuesta'
            }), 500
        
        return jsonify({
            'success': True,
            'response': response.text
        })
    
    except Exception as e:
        app.logger.error(f"Error en /chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error al procesar tu mensaje. Por favor, intenta de nuevo.'
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
