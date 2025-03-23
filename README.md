# Customer-Churn-Prediction

Overview:
This project is a Customer Churn Prediction System that identifies customers who are likely to leave a company. The system uses Machine Learning models (Logistic Regression & Random Forest) to analyze customer behavior based on demographic and financial attributes.

Features:
Loads and preprocesses customer dataset
Handles missing values and encodes categorical features
Trains Logistic Regression & Random Forest classifiers
Evaluates model performance using accuracy, precision, recall, and ROC-AUC score
Provides feature importance insights

Dataset:
The dataset (Churn_Modelling.csv) consists of customer records with various attributes:

RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
1,15634602,Hargrave,619,France,Female,42,2,0.00,1,1,1,101348.88,1
2,15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
3,15619304,Onio,502,France,Female,42,8,159660.80,3,1,0,113931.57,1

CustomerId, Surname, RowNumber → Irrelevant columns (dropped during preprocessing)

CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary → Features

Exited → Target variable (1 = Churned, 0 = Retained)

Download the dataset from https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

Installation & Setup:
1. Clone the Repository

git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Install Dependencies
Make sure you have Python installed, then install required libraries:

pip install pandas numpy scikit-learn

Running the Project:
Place the dataset (Churn_Modelling.csv) in the project folder.

Run the Python script:

python customer_churn.py

Model Training & Evaluation:
Preprocessing: Handles missing values, encodes categorical features, and scales numerical data.
Feature Selection: Drops irrelevant columns and selects useful features.
Training: Trains Logistic Regression & Random Forest classifiers.
Evaluation: Measures accuracy, confusion matrix, and ROC-AUC score.
Feature Importance: Analyzes which attributes influence churn the most.

Sample Output:

Model: LogisticRegression
Accuracy Score: 81%
Confusion Matrix:
[[1593  407]
 [ 289  211]]
Classification Report:
Precision: 0.72, Recall: 0.57, F1-score: 0.64
ROC AUC Score: 0.76

Feature Importance (Top 5):
Balance            0.27
Age               0.21
NumOfProducts     0.18
CreditScore       0.13
IsActiveMember    0.12

