import pandas as pd
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATASETS
# ==========================================

train_data = pd.read_csv("train_all.csv")
val_data = pd.read_csv("val_all.csv")
test_data = pd.read_csv("test_all.csv")

print("Datasets loaded successfully!")

print("\nTraining dataset shape:")
print(train_data.shape)

print("\nValidation dataset shape:")
print(val_data.shape)

print("\nTesting dataset shape:")
print(test_data.shape)


# ==========================================
# 2. PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove stop words
    text = " ".join(
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    )

    # Lemmatization
    text = " ".join(
        Word(word).lemmatize()
        for word in text.split()
    )

    return text


# ==========================================
# 3. PREPROCESS ALL DATASETS
# ==========================================

print("\nPreprocessing training data...")

train_data["clean_text"] = train_data["sentence"].astype(str).apply(
    preprocess_text
)

print("Preprocessing validation data...")

val_data["clean_text"] = val_data["sentence"].astype(str).apply(
    preprocess_text
)

print("Preprocessing testing data...")

test_data["clean_text"] = test_data["sentence"].astype(str).apply(
    preprocess_text
)


print("\nExample of preprocessing:")

print(
    train_data[
        ["sentence", "clean_text", "label"]
    ].head()
)


# ==========================================
# 4. SEPARATE TEXT AND LABELS
# ==========================================

X_train_text = train_data["clean_text"]
y_train = train_data["label"]

X_val_text = val_data["clean_text"]
y_val = val_data["label"]

X_test_text = test_data["clean_text"]
y_test = test_data["label"]


# ==========================================
# 5. CREATE TF-IDF FEATURES
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000
)


# IMPORTANT:
# Fit TF-IDF ONLY on training data

X_train = vectorizer.fit_transform(
    X_train_text
)

X_val = vectorizer.transform(
    X_val_text
)

X_test = vectorizer.transform(
    X_test_text
)


print("\nTF-IDF shapes:")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==========================================
# 6. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 7. VALIDATION
# ==========================================

print("\n================================")
print("VALIDATION PERFORMANCE")
print("================================")

val_predictions = model.predict(X_val)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

print("Validation Macro F1 Score:", val_f1)


# ==========================================
# 8. FINAL TESTING
# ==========================================

print("\n================================")
print("FINAL TEST PERFORMANCE")
print("================================")

test_predictions = model.predict(X_test)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("Test Macro F1 Score:", test_f1)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[
        "negative",
        "neutral",
        "positive"
    ]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "negative",
        "neutral",
        "positive"
    ],
    yticklabels=[
        "negative",
        "neutral",
        "positive"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Sentiment Classification Confusion Matrix"
)

plt.savefig(
    "confusion_matrix_large.png"
)

plt.close()

print("\nConfusion matrix saved as:")
print("confusion_matrix_large.png")


# ==========================================
# 11. TEST NEW SENTENCES
# ==========================================

new_sentences = [
    "I absolutely loved this movie",
    "This movie was terrible",
    "The movie was okay",
    "The acting was amazing",
    "I really disliked this film"
]


new_sentences_clean = [
    preprocess_text(sentence)
    for sentence in new_sentences
]


# Convert new sentences into TF-IDF

new_X = vectorizer.transform(
    new_sentences_clean
)


# Predict sentiment

new_predictions = model.predict(
    new_X
)


print("\n================================")
print("NEW SENTENCE PREDICTIONS")
print("================================")


for sentence, prediction in zip(
    new_sentences,
    new_predictions
):

    print(
        sentence,
        "->",
        prediction
    )