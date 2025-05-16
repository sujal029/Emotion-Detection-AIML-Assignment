import pandas as pd
import neattext as nt

df = pd.read_csv("emotions.csv")

# Clean text column
df['clean_text'] = df['text'].apply(lambda x: nt.TextCleaner(x).remove_special_characters().text.lower())

# Drop rows where either clean_text or emotion is missing
df = df.dropna(subset=['clean_text', 'emotion'])

df.to_csv("emotions_clean.csv", index=False)
print("✅ Text cleaned and saved as emotions_clean.csv")
