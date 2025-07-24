from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, vectorizer, X_test, y_test):
    # Transform test data using the same vectorizer
    X_test_vec = vectorizer.transform(X_test)

    # Predict using trained model
    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))