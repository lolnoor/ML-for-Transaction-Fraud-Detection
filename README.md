# ML-for-Transaction-Fraud-Detection
Overview
This repository contains code that demonstrates the use of machine learning techniques for transaction monitoring in anti-money laundering (AML) systems. The goal is to leverage machine learning algorithms to identify suspicious transactions that could potentially indicate money laundering activities.

Features
Use of Random Forest algorithm for classification.
Focus on detecting abnormal or suspicious patterns in transaction data.
Use of unsupervised learning techniques like Autoencoders for anomaly detection.
Explanation of the steps from data preprocessing to model evaluation.
Project Structure: Your folder structure might look like this:
aml_project/
├── aml-env/                 #virtual environment folder
├── aml_transaction_monitoring.py   #Python script
Running the Script:

With the virtual environment activated, run your Python script: python aml_transaction_monitoring.py
Code Structure
aml_transaction_monitoring.py: Contains the full machine learning pipeline including data preprocessing, feature engineering, model training, and evaluation.
Requirements
To run this code, you need the following Python packages:

pandas
numpy
scikit-learn
keras
matplotlib
Install the dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn keras matplotlib
How to Use
Clone the repository:
bash
Copy code
git clone https://github.com/lolnoor/ML-for-Transaction-Fraud-Detection.git
cd ML-for-Transaction-Fraud-Detection
Prepare the data:

The code expects the data to be in a CSV format. Make sure you have a dataset that includes transaction information such as transaction amount, date, account details, etc.
Modify the load_data() function in aml_ml_model.py to point to your dataset.
Run the script:

bash
Copy code
python aml_ml_model.py
Evaluate the model:
The script will output the performance metrics such as accuracy, precision, recall, and F1-score.
It will also plot feature importance and other visualizations related to model performance.
Code Highlights
Random Forest for Classification
python
Copy code
from sklearn.ensemble import RandomForestClassifier

# Initializing the model
rf_model = RandomForestClassifier(n_estimators=100)

# Fitting the model
rf_model.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf_model.predict(X_test)
Autoencoders for Anomaly Detection
python
Copy code
from keras.models import Model
from keras.layers import Input, Dense

# Building the Autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(14, activation="tanh")(input_layer)
encoder = Dense(7, activation="relu")(encoder)
decoder = Dense(14, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile and fit the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True)
Evaluation
Once the model is trained, you can evaluate it using various performance metrics:

python
Copy code
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
Contributions
Feel free to open issues or submit pull requests if you want to improve the code or add new features.

License
This project is licensed under the MIT License. 

