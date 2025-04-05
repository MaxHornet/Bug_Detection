import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb # type: ignore
import joblib

# Step 1: Prepare dataset
buggy_code_samples = [
    ("print(Hello World)", 'print("Hello World")', "SyntaxError"),
    ("for i in range(5)\nprint(i)", "for i in range(5):\n    print(i)", "IndentationError"),
    ("if x == 5 print('Hello')", "if x == 5:\n    print('Hello')", "SyntaxError"),
    ("def foo()\n  return 5", "def foo():\n    return 5", "SyntaxError"),
    ("x = 5 + '3'", "x = 5 + int('3')", "TypeError"),
    ("result = 'hello' * [2]", "result = 'hello' * 2", "TypeError"),
    ("val = len(5)", "val = len(str(5))", "TypeError"),
    ("x = 'hello' + 5", "x = 'hello' + str(5)", "TypeError"),
    ("print(x)", "x = 10\nprint(x)", "NameError"),
    ("total = price * quantity", "price, quantity = 10, 5\ntotal = price * quantity", "NameError"),
    ("my_function()", "def my_function():\n    return 42\nmy_function()", "NameError"),
    ("print(unknown_var)", "unknown_var = 10\nprint(unknown_var)", "NameError"),
    ("def foo():\nprint('Hello')", "def foo():\n    print('Hello')", "IndentationError"),
    ("if True:\nprint('Indented')", "if True:\n    print('Indented')", "IndentationError"),
    ("for i in range(3):\nprint(i)", "for i in range(3):\n    print(i)", "IndentationError"),
    ("arr = [1, 2, 3]\nprint(arr[5])", "arr = [1, 2, 3]\nprint(arr[-1])", "IndexError"),
    ("empty_list = []\nprint(empty_list[0])", "if empty_list:\n    print(empty_list[0])", "IndexError"),
    ("my_dict = {'a': 1}\nprint(my_dict['b'])", "print(my_dict.get('b', 'Key not found'))", "KeyError"),
    ("config = {'host': 'localhost'}\nprint(config['port'])", "print(config.get('port', 'default_port'))", "KeyError"),
    ("result = 10 / 0", "result = 10 / (1 if 0 else 1)", "ZeroDivisionError"),
    ("val = 5 % 0", "val = 5 % (1 if 0 else 1)", "ZeroDivisionError"),
] * 20  # Repeat to get 500+ samples 

df = pd.DataFrame(buggy_code_samples, columns=["BuggyCode", "FixedCode", "BugType"])

# Step 2: Feature extraction
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3))
X = vectorizer.fit_transform(df["BuggyCode"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["BugType"])

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Train the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = xgb_model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 6: Save model and encoders
joblib.dump(xgb_model, "bug_classifier_xgb.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\n💾 Model, vectorizer, and label encoder saved successfully!")

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
