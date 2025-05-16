import joblib

# Load the trained pipeline (vectorizer + model)
pipeline = joblib.load('emotion_model.pkl')

print("\nEnter a sentence (or type 'exit' to quit): ")
while True:
    text = input("> ")
    if text.lower() == 'exit':
        break
    cleaned_text = text.lower()  # Basic cleaning for prediction
    prediction = pipeline.predict([cleaned_text])[0]
    print("🔮 Predicted Emotion: ", prediction)
