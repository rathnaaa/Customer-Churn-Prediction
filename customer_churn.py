import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# 1. Load the Data
def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        print("Data Loaded Successfully!")
        print("Columns in dataset:", data.columns)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# 2. Preprocess the Data
def preprocess_data(data):
    """
    Handles missing values, encodes categorical features, and scales numerical features.
    """
    # Drop irrelevant columns if needed (Example: customer ID)
    if 'CustomerID' in data.columns:
        data.drop(columns=['CustomerID'], inplace=True)
    
    # --- Handle Missing Values ---
    imputer = SimpleImputer(strategy="most_frequent")  # Can use 'mean' for numerical data
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # --- Encode Categorical Features ---
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store encoders if needed for inference
    
    # --- Convert data types back (Imputer changes everything to object) ---
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')
    
    return data

# 3. Select Features & Target
def select_features(data, target_column='Churn'):  # Adjust target column name if needed
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# 4. Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# 5. Train Models (Logistic Regression & Random Forest)
def train_models(X_train, y_train):
    """
    Trains both Logistic Regression and RandomForest models.
    """
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# 6. Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using Accuracy, Confusion Matrix, and ROC-AUC score.
    """
    predictions = model.predict(X_test)
    print(f"\nModel: {model.__class__.__name__}")
    print("Accuracy Score:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# --- Main Execution ---
filepath = "C:/Users/rathn/OneDrive/ドキュメント/GrowthLink/dataset_customerchurn/Churn_Modelling.csv"  # Change to your file path

data = load_data(filepath)
if data is None:
    exit()

# Preprocess the data
data = preprocess_data(data)

# Select features and target
X, y = select_features(data, target_column='Exited')  # Change 'Exited' if dataset has another churn indicator

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Scale features AFTER splitting to avoid data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
models = train_models(X_train, y_train)

# Evaluate models
for model_name, model in models.items():
    evaluate_model(model, X_test, y_test)

# Feature Importance (Only for Random Forest)
rf_model = models["Random Forest"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance (Top 10):\n", feature_importance.head(10))
