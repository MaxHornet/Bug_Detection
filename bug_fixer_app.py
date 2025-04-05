import streamlit as st
import joblib

# Load the model and vectorizer
xgb_model = joblib.load("bug_classifier_xgb.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Bug type to label
bug_classes = {
    0: "IndentationError",
    1: "IndexError",
    2: "KeyError",
    3: "NameError",
    4: "SyntaxError",
    5: "TypeError",
    6: "ZeroDivisionError"
}

# Fix logic based on predicted bug
def fix_code(buggy_code, bug_type):
    if bug_type == "SyntaxError":
        return buggy_code.replace('print(Hello World)', 'print("Hello World")')
    elif bug_type == "IndentationError":
        return buggy_code.replace("for i in range(5)\nprint(i)", "for i in range(5):\n    print(i)")
    elif bug_type == "ZeroDivisionError":
        return buggy_code.replace("10 / 0", "10 / (1 if 0 == 0 else 1)")
    elif bug_type == "IndexError":
        return buggy_code.replace("print(arr[5])", "print(arr[min(5, len(arr) - 1)])")
    elif bug_type == "KeyError":
        return buggy_code.replace("print(my_dict['b'])", "print(my_dict.get('b', 'Key not found'))")
    elif bug_type == "NameError":
        return "price, quantity = 1, 1\n" + buggy_code
    elif bug_type == "TypeError":
        return buggy_code.replace("5 + '3'", "5 + str(3)")
    return buggy_code

# Streamlit UI
st.set_page_config(page_title="🐞 Bug Detection and Fixing", layout="centered")
st.title("🐞 Bug Detection and Fixing")

code_input = st.text_area("Paste your buggy Python code here:", height=200)

if st.button("🔍 Detect and Fix"):
    if code_input.strip() == "":
        st.warning("Please enter some Python code.")
    else:
        # Predict bug type
        input_vector = vectorizer.transform([code_input])
        prediction = xgb_model.predict(input_vector)[0]
        bug_type = label_encoder.inverse_transform([prediction])[0]

        # Fix code
        fixed_code = fix_code(code_input, bug_type)

        st.markdown(f"### 🚨 Predicted Bug Type: `{bug_type}`")
        st.markdown("### 🛠️ Suggested Fix:")
        st.code(fixed_code, language="python")
