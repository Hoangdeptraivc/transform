from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_llama

app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://localhost:63342"}})
user_waiting_info = {}
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_message = request.json.get('message')
    reply = ask_llama(user_message)

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
