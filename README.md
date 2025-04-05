Live website for this Project ~ https://bugdetection-lkgd2app3u2pfb4ydapp9ul8.streamlit.app/ 

And if you want to manually try the project out then >

Steps to Test the Bug Detection and Fixing Model Locally

1. Install the repo files to a local folder

Clone or download the entire GitHub repository into a folder on your system.



2. Open the folder in VS Code

Launch Visual Studio Code and open the project folder.



3. Create a virtual environment

python -m venv venv


4. Activate the virtual environment

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate



5. Install required packages

pip install streamlit xgboost scikit-learn pandas joblib


6. Run the training script

This will train the model and generate the .pkl files used for bug detection:


python train_model.py


7. Evaluate the model

Run the testing script to get model accuracy:


python test_model.py


8. Launch the web interface

This runs the Streamlit web UI:


streamlit run bug_fixer_app.py


9. Test using sample inputs

Use the provided samples.txt file to manually input buggy code snippets into the web interface and see predictions and fixed code.
