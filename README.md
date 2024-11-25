# Adaptive-Learner
Adaptive Learning Application for Learning Middle-School English

## Instructions to Run/Test Code
1. Clone the repository.
2. Create and activate a python virtual environment and install the required modules from `requirements.txt`.

   ```
   python -m venv venv
   source venv/bin/activate
   pip install requirements.txt
   ```
   
4. To train and test the model, please generate a hugging face token and add it to the environment as `HF_TOKEN`. You can use the following command.
   ```
   export HF_TOKEN="your-token-here"
   ```
   To train, run `model.py` file. 
   ```
   python model.py
   ```
   To test, run `test.py` file.
   ```
   python test.py
   ```
   
5. To check the web application built using the GPT4All model,

   ```
   TO-BE-UPDATED-SOON.
   ```
