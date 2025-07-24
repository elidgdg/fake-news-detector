from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)
model, vectorizer = train_model(X_train, y_train)
evaluate_model(model, vectorizer, X_test, y_test) 