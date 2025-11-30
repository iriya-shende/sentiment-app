import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import streamlit as st


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()


st.title("📱 Sentiment Analysis App")
st.write("Analyze any review as **Positive** or **Negative**")


user_input = st.text_area("Enter a review:", height=150)


if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please type something.")
    else:
        cleaned = preprocess(user_input)
        X = vectorizer.transform([cleaned])
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]


        label = "Positive 😊" if pred == 1 else "Negative 😠"
        confidence = proba[pred]


        st.subheader("Prediction")
        st.success(f"{label} (Confidence: {confidence:.2f})")