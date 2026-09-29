'''python'''
import pandas as pd
import string

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("dataset.csv")

print("Original Dataset:")
print(data)


# -----------------------------------
# 2. Convert Text to Lowercase
# -----------------------------------

data["text"] = data["text"].str.lower()

print("\nAfter Lowercasing:")
print(data)


# -----------------------------------
# 3. Remove Punctuation
# -----------------------------------

data["text"] = data["text"].apply(
    lambda x: x.translate(
        str.maketrans("", "", string.punctuation)
    )
)

print("\nAfter Removing Punctuation:")
print(data)


# -----------------------------------
# 4. Remove Stop Words
# -----------------------------------

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data["text"] = data["text"].apply(
    lambda x: " ".join(
        word for word in x.split()
        if word not in ENGLISH_STOP_WORDS
    )
)

print("\nAfter Removing Stop Words:")
print(data)


# -----------------------------------
# 5. Lemmatization
# -----------------------------------

from textblob import Word

data["text"] = data["text"].apply(
    lambda x: " ".join(
        Word(word).lemmatize()
        for word in x.split()
    )
)

print("\nAfter Lemmatization:")
print(data)


# -----------------------------------
# 6. Convert Text into Numbers using TF-IDF
# -----------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["text"])

print("\nTF-IDF Matrix:")
print(X.toarray())


# -----------------------------------
# 7. Separate Features and Labels
# -----------------------------------

y = data["sentiment"]


# -----------------------------------
# 8. Split Data into Training and Testing
# -----------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 9. Create Logistic Regression Model
# -----------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)


# -----------------------------------
# 10. Make Predictions
# -----------------------------------

y_pred = model.predict(X_test)

print("\nPredicted:", y_pred)
print("Actual:", y_test.values)


# -----------------------------------
# 11. Calculate F1 Score
# -----------------------------------

from sklearn.metrics import f1_score

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nF1 Score:", f1)


# -----------------------------------
# 12. Create Confusion Matrix
# -----------------------------------

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
)


# -----------------------------------
# 13. Display Confusion Matrix
# -----------------------------------

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["negative", "neutral", "positive"],
    yticklabels=["negative", "neutral", "positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

# -----------------------------------
# 14. Test the Model with New Sentences
# -----------------------------------

new_sentences = [
    "I loved this film",
    "This movie was terrible",
    "The movie was okay"
]

# Apply the same preprocessing
new_sentences_clean = [
    sentence.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    for sentence in new_sentences
]

new_sentences_clean = [
    " ".join(
        word for word in sentence.split()
        if word not in ENGLISH_STOP_WORDS
    )
    for sentence in new_sentences_clean
]

new_sentences_clean = [
    " ".join(
        Word(word).lemmatize()
        for word in sentence.split()
    )
    for sentence in new_sentences_clean
]

    # -----------------------------------
# 14. Test the Model with New Sentences
# -----------------------------------

new_sentences = [
    "I loved this film",
    "This movie was terrible",
    "The movie was okay"
]

# Preprocess the new sentences
new_sentences_clean = [
    sentence.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    for sentence in new_sentences
]

new_sentences_clean = [
    " ".join(
        word for word in sentence.split()
        if word not in ENGLISH_STOP_WORDS
    )
    for sentence in new_sentences_clean
]

new_sentences_clean = [
    " ".join(
        Word(word).lemmatize()
        for word in sentence.split()
    )
    for sentence in new_sentences_clean
]

# Convert new sentences into TF-IDF
new_X = vectorizer.transform(new_sentences_clean)

# Predict sentiment
new_predictions = model.predict(new_X)

print("\nNew Sentence Predictions:")

for sentence, prediction in zip(new_sentences, new_predictions):
    print(sentence, "->", prediction)

    large_data = pd.read_csv("train_all.csv")

print(large_data.head())
print(large_data["label"].value_counts())