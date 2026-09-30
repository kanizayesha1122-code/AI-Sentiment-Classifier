# AI Sentiment Classifier

## 📌 Project Overview

This project is an **Intelligent Multi-Class Natural Language Text Sentiment Classifier** developed as part of an AI internship project.

The system uses **Natural Language Processing (NLP)** and **Machine Learning** to classify text into three sentiment categories:

* 🔴 Negative
* 🟡 Neutral
* 🟢 Positive

The project demonstrates the complete machine learning pipeline, from text preprocessing and feature extraction to model training, evaluation, and prediction on new sentences.

---

## 🎯 Objectives

The main objectives of this project are to:

* Preprocess natural language text
* Remove stop words
* Apply lemmatization
* Convert text into numerical features using TF-IDF
* Train a multi-class sentiment classification model
* Evaluate the model using F1-score and a confusion matrix
* Test the trained model on new sentences

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **NLTK**
* **TextBlob**
* **Matplotlib**
* **Seaborn**
* **TF-IDF**
* **Logistic Regression**

---

## 🔄 NLP Pipeline

The project follows this workflow:

**Raw Text → Preprocessing → TF-IDF → Logistic Regression → Sentiment Prediction**

### 1. Text Preprocessing

The input text is processed using:

* Lowercasing
* Punctuation removal
* Stop-word removal
* Lemmatization

For example:

```text
"I absolutely loved this movie!"
```

is cleaned before being converted into numerical features.

### 2. TF-IDF Feature Extraction

The cleaned text is converted into numerical vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

The final vectorizer uses a maximum of **10,000 features**.

### 3. Classification

A **Logistic Regression** classifier is trained on the TF-IDF features to predict one of the three sentiment classes:

```text
negative
neutral
positive
```

---

## 📊 Dataset

The project uses a larger multi-class sentiment dataset divided into:

* Training set: **102,097 samples**
* Validation set: **5,421 samples**
* Test set: **6,530 samples**

The dataset files are kept locally and are excluded from GitHub using `.gitignore`.

---

## 📈 Model Evaluation

The model was evaluated on the provided test set.

### Final Test Results

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    |   0.59 |
| Macro F1    | 0.5863 |
| Weighted F1 |   0.59 |

### Classification Report

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Negative |      0.70 |   0.41 |     0.52 |
| Neutral  |      0.47 |   0.79 |     0.59 |
| Positive |      0.69 |   0.62 |     0.65 |

The validation Macro F1-score was **0.5816**, while the final test Macro F1-score was **0.5863**.

---

## 📉 Confusion Matrix

The project generates a confusion matrix to visualize the model's predictions across the three sentiment classes.

The resulting visualization is included in this repository as:

`confusion_matrix_large.png`

---

## 🧪 Example Predictions

The trained model was tested on new sentences:

| Input                         | Prediction |
| ----------------------------- | ---------- |
| I absolutely loved this movie | Positive   |
| This movie was terrible       | Negative   |
| The movie was okay            | Positive   |
| The acting was amazing        | Positive   |
| I really disliked this film   | Negative   |

These examples demonstrate that the trained model can perform sentiment predictions on previously unseen text.

---

## 📁 Project Structure

```text
AI-Sentiment-Classifier/
│
├── sentiment_classifier_large.py
├── confusion_matrix_large.png
├── dataset.csv
├── .gitignore
└── README.md
```

The larger dataset files are intentionally excluded from the repository through `.gitignore`.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/kanizayesha1122-code/AI-Sentiment-Classifier.git
```

### 2. Open the project folder

```bash
cd AI-Sentiment-Classifier
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install pandas scikit-learn nltk textblob matplotlib seaborn
```

### 6. Run the classifier

```bash
python sentiment_classifier_large.py
```

---

## ⚠️ Limitations

This project uses a **TF-IDF + Logistic Regression baseline**, so it does not understand language in the same way as modern transformer-based NLP models.

The test results also show that performance differs between sentiment classes. For example, the model achieved a recall of **0.79 for neutral text** but **0.41 for negative text**.

Possible future improvements include:

* Using word and bigram features
* Hyperparameter tuning
* Class-weight optimization
* Trying alternative machine learning models
* Using word embeddings
* Experimenting with transformer-based NLP models

---

## 🚀 Future Improvements

Future versions of the project could explore more advanced NLP techniques such as:

* Word embeddings
* Recurrent Neural Networks
* LSTM networks
* BERT and other transformer models
* Hyperparameter optimization
* Larger and more domain-specific datasets

---

## 👩‍💻 Author

**Kaniz Ayesha**

AI Undergraduate Student
University of Peshawar

---

## 📌 Internship Project

Developed as part of the **Progree Remote Internship Program – AI Internship**.
