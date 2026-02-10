import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Fitting TF-IDF...")

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])   # THIS LINE FITS TF-IDF
y = df['label']

print("Training model...")

model = MultinomialNB()
model.fit(X, y)

print("Saving files...")

pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ DONE SUCCESSFULLY")
