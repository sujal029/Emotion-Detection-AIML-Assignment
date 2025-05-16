import pandas as pd
df = pd.read_csv("C:\Users\baiss\OneDrive\Desktop\portfolio\emotion_detection_project\emotions.csv")
print(df.isnull().sum())
