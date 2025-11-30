📱 Sentiment Analysis Web App

Classifies Amazon reviews as Positive or Negative using Machine Learning.

🔗 Live App: https://sentiment-app-dkiyuxacvy8eevxnd3gw77.streamlit.app

🔗 GitHub Repo: https://github.com/iriya-shende/sentiment-app

⭐ About the Project

This is a complete end-to-end Sentiment Analysis project.
Users can type any review and the app predicts if the sentiment is Positive or Negative with a confidence score.

🧠 How It Works

Cleans the text (punctuation, numbers, stopwords)

Tokenizes & lemmatizes using NLTK

Converts text to vectors using CountVectorizer

Uses Logistic Regression for prediction

Shows result instantly on Streamlit web app

📂 Files in the project

app.py → Streamlit web app

train_model.py → Trains the ML model

model.pkl → Saved ML model

vectorizer.pkl → Saved vectorizer

amazon_cells_labelled.txt → Dataset

requirements.txt → Dependencies

🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

✨ Tools Used

Python, Scikit-learn, NLTK, Streamlit, Joblib, GitHub, Google Colab
