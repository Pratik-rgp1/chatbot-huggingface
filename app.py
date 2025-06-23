from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress future warnings (from huggingface_hub)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the pre-trained BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    user_input = data['prompt'].strip()

    # Format conversation history + user input
    history_text = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_text, user_input, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length= 60)  # max_length will cause the model to crash at some point as history grows

    # Decode and display response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    # Update conversation history
    conversation_history.append(user_input)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run()