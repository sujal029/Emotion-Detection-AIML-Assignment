from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('emotion_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
