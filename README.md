# Bug_Detection and Fixing Project
For locally testing the model follow the steps:

1. First install the repo files to a disk folder 
2. open the folder in vscode for testing the model and create a new terminal
3. create a virtual environment : python -m venv venv
4. activate the environment : venv\Scripts\activate
5. Install Required Packages : pip install streamlit xgboost scikit-learn pandas joblib
6. Run the train model script , which will train the model and generate the pkl files which will be used for detecting the bugs: train_model.py
7. Then evaluate the model by running the test script : test_model.py
8. Now test the model which is integrated with the webui : streamlit run bug_fixer_app.py
9. A samples.txt file is provided for manually inputting the codes into the webui for testing it.
