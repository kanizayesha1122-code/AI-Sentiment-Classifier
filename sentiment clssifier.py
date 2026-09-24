import pandas as pd

data = pd.read_csv("dataset.csv")

print(data)
data["text"] = data["text"].str.lower()

print(data)
import string

data["text"] = data["text"].apply(
    lambda x: x.translate(str.maketrans("", "", string.punctuation))
)

print(data)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data["text"] = data["text"].apply(
    lambda x: " ".join(
        word for word in x.split()
        if word not in ENGLISH_STOP_WORDS
    )
)

print(data)
from textblob import Word

data["text"] = data["text"].apply(
    lambda x: " ".join(
        Word(word).lemmatize()
        for word in x.split()
    )
)
print(data)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["text"])

print(X.toarray())