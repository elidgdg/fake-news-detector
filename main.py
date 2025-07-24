from data_loader import load_data
from preprocessing import preprocess

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)