📰 Fake News Detection | NLP & Machine Learning

A complete Fake News Classification Project that identifies whether a news article is Real or Fake using Natural Language Processing (NLP) techniques and a Logistic Regression model. The model is deployed with a Streamlit web app (fake.py) for real-time predictions.


---

📌 Project Overview

This project focuses on detecting fake news by analyzing the text content of news articles using machine learning and NLP.
It includes:
✔ Fake & Real news datasets (fake.csv, true.csv)
✔ Text preprocessing using NLTK (stopwords, PorterStemmer)
✔ TF-IDF Vectorization for text to numeric conversion
✔ Model training using Logistic Regression
✔ Evaluation using accuracy_score & classification_report
✔ Saved trained model & vectorizer using Joblib
✔ Streamlit-based web application (fake.py) for deployment


---

🛠 Tech Stack

Component	Technology Used

Language	Python
NLP Tools	NLTK (stopwords, PorterStemmer)
Feature Extraction	TfidfVectorizer
ML Algorithm	Logistic Regression
Evaluation Metrics	accuracy_score, classification_report
Model Saving	Joblib (vectorizer.jb, logistic.jb)
Deployment	Streamlit (fake.py)
Dataset	Fake.csv, True.csv



---

📂 Dataset Description

File Name	Description

Fake.csv	Contains fake news articles
True.csv	Contains real news articles


Both datasets are merged and preprocessed before model training.


---

⚙ End-to-End Workflow

✅ 1. Data Loading

import pandas as pd
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

✅ 2. Text Preprocessing (NLTK)

Convert to lowercase

Remove punctuation & symbols

Remove stopwords

Apply PorterStemmer


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

✅ 3. Feature Extraction

TF-IDF converts textual data to numerical format:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

Saved as: vectorizer.jb

✅ 4. Model Training

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

Saved as: logistic.jb

✅ 5. Model Evaluation

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


---

💾 Model Saving

import joblib
joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(model, "logistic.jb")


---

🚀 Streamlit App (Deployment)

Run the web app using:

streamlit run fake.py

Streamlit Features:

✔ Input news text
✔ Loads vectorizer.jb & logistic.jb
✔ Predicts: ✅ Real or ❌ Fake
✔ Simple and interactive UI


---

📊 Results

Metric	Score

Accuracy	98% 
Precision: 99% 	
Recall: 99%


---

🧠 Key Learnings

✅ Text preprocessing using NLTK
✅ TF-IDF based feature engineering
✅ Logistic Regression for binary classification
✅ Saving/loading model with Joblib
✅ Deploying ML model using Streamlit


---

🔮 Future Enhancements

Use LSTM / Bidirectional LSTM / Transformers

Add News Title + Author + Subject as features

Deploy on cloud (Heroku / AWS / Render)

Add visual analytics dashboard



---


✍ Author

👤 Kousik Chakraborty
📧 Email: www.kousik.c.in@gmail.com
🔗 GitHub Profile: https://github.com/iamkousikc-create18
🔗 Project Repository: https://github.com/iamkousikc-create18/FakeOrRealNews
