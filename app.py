import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK files
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Text Preprocessing
# --------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# --------------------------
# Train Model
# --------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv(
        "twitter_training.csv",
        header=None
    )

    df.columns = [
        "TweetID",
        "Entity",
        "Sentiment",
        "Tweet"
    ]

    df.dropna(inplace=True)

    df["Tweet"] = df["Tweet"].apply(clean_text)

    X = df["Tweet"]
    y = df["Sentiment"]

    tfidf = TfidfVectorizer(
        max_features=5000
    )

    X = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )

    return model, tfidf, accuracy

model, tfidf, accuracy = train_model()

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="😊"
)

st.title(
    "Twitter Sentiment Analysis using NLP + TF-IDF + Naive Bayes"
)

st.write(
    "Dataset: Kaggle Twitter Sentiment Dataset"
)

st.success(
    f"Model Accuracy: {accuracy*100:.2f}%"
)

user_input = st.text_area(
    "Enter Tweet / Review"
)

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning(
            "Please enter some text."
        )

    else:

        cleaned = clean_text(
            user_input
        )

        vector = tfidf.transform(
            [cleaned]
        )

        prediction = model.predict(
            vector
        )[0]

        st.subheader(
            "Prediction"
        )

        if prediction.lower() == "positive":
            st.success(
                "😊 Positive"
            )

        elif prediction.lower() == "negative":
            st.error(
                "😞 Negative"
            )

        elif prediction.lower() == "neutral":
            st.info(
                "😐 Neutral"
            )

        else:
            st.warning(
                f"⚪ {prediction}"
            )

        st.subheader(
            "Processed Text"
        )

        st.write(
            cleaned
        )
