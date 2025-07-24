from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    # Convert text to TF-IDF vectors
    vectorizer = TfidVectorizer(max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer