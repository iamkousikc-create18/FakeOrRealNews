📰 Fake News Detection using Logistic Regression

📘 Overview

This project detects Fake News using Machine Learning (Logistic Regression). It applies Natural Language Processing (NLP) to clean and analyze news text, and predicts whether a given news article is Real or Fake.

The trained model is deployed using Streamlit, providing a simple web interface for real-time testing.


---

🚀 Features

Clean text using regular expressions and NLP preprocessing

Train model using Logistic Regression

Achieved 98% accuracy on the dataset

Deployable Streamlit web app for interactive fake news detection



---

🧠 Technologies Used

Python

pandas, numpy, scikit-learn

nltk, re

joblib (for saving/loading models)

Streamlit (for deployment)



---

⚙ How It Works

1. Text Cleaning: Removes punctuation, URLs, and unwanted symbols.


2. Vectorization: Converts text into numerical features using TfidfVectorizer.


3. Model Training: Logistic Regression classifier is trained on the processed data.


4. Prediction: The trained model predicts whether a given news article is Real or Fake.


5. Deployment: Streamlit app lets users input news text and see instant results.




---

🖥 Streamlit App Usage

Run the following command to start the app:

streamlit run fake.py

Then enter any news article text into the input box — the app will tell you if it’s Real ✅ or Fake ❌.


---

📁 Project Structure

├── NewsFakeOrReal.ipynb           # Model training and preprocessing notebook
├── fake.py              # Streamlit web app for deployment
├── logistic.jb          # Trained Logistic Regression model
├── vectorizer.jb        # TF-IDF vectorizer
├── requirements.txt     # Project dependencies
├── readme.md            # Documentation
└── True.csv             # True News dataset
└── Fake.csv             # Fake News dataset

---

📊 Model Performance

Algorithm: Logistic Regression

Accuracy: 98%

Evaluation: Tested on unseen data for validation



---

🔚 Conclusion

This project successfully demonstrates how Machine Learning and NLP can be combined to build a powerful Fake News Detection System. The Streamlit web app makes it easy for users to verify the authenticity of news in real-time.

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
