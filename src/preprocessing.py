import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()

    # Remove punctuation and digits
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)

    # Remove stopwords ("the", "a", "and", etc.)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)

def preprocess(df):
    # Clean 'text' column
    df["clean_text"] = df["text"].apply(clean_text)

    # Split features and labels
    X = df["clean_text"]
    y = df["label"]

    # Split into training and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)