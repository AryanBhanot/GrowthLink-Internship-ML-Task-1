import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_tfidf_features(corpus):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
