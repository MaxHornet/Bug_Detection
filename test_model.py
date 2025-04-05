import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and necessary objects
xgb_model = joblib.load("bug_classifier_xgb.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Sample test data with true labels
test_data = [
    ("print(Hello World)", "SyntaxError"),
    ("for i in range(5)\nprint(i)", "IndentationError"),
    ("x = 10 / 0", "ZeroDivisionError"),
    ("arr = [1, 2, 3]\nprint(arr[5])", "IndexError"),
    ("my_dict = {'a': 1}\nprint(my_dict['b'])", "KeyError"),
    ("total = price * quantity", "NameError"),
    ("x = 5 + '3'", "TypeError")
]

# Separate code and labels
X_test_raw = [code for code, label in test_data]
y_true_labels = [label for code, label in test_data]

# Vectorize test input and encode labels
X_test = vectorizer.transform(X_test_raw)
y_true = label_encoder.transform(y_true_labels)

# Predict with the model
y_pred = xgb_model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Show individual predictions
for i in range(len(test_data)):
    print(f"\n🔍 Code:\n{X_test_raw[i]}")
    print(f"✅ Actual: {y_true_labels[i]}")
    print(f"🚨 Predicted: {y_pred_labels[i]}")

# Print accuracy score and classification report
print("\n📊 Accuracy:", accuracy_score(y_true, y_pred))
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
