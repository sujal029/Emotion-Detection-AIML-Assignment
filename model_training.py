import pandas as pd
import neattext as nt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("emotions.csv") # Make sure the CSV is in the same folder
df['emotion'] = df['emotion'].str.lower().str.strip()

# Clean text
df['clean_text'] = df['text'].apply(lambda x: nt.TextCleaner(x).remove_special_characters().text.lower())

# Split data
X = df['clean_text']
y = df['emotion']

# Pipeline for TF-IDF and Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, 'emotion_model.pkl')
print("✅ Model saved as 'emotion_model.pkl'")
