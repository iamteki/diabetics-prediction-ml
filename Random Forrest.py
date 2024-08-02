import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diabetes dataset into a pandas DataFrame
file_path = "cleanedDataset.csv"
diabetes_dataset = pd.read_csv(file_path)

# Handle missing values by imputation
imputer = SimpleImputer(strategy='mean')
diabetes_dataset = pd.DataFrame(imputer.fit_transform(diabetes_dataset), columns=diabetes_dataset.columns)

# Separate features (X) and target variable (Y)
X = diabetes_dataset.drop(columns=['Class Label(GDM /Non GDM)', 'Case Number'], axis=1)
Y = diabetes_dataset['Class Label(GDM /Non GDM)']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train the Random Forest classifier using GridSearchCV for hyperparameter tuning
rf_classifier = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, refit=True, verbose=2)
rf_classifier.fit(X_train, Y_train)

# Make predictions on training and testing data
Y_train_pred = rf_classifier.predict(X_train)
Y_test_pred = rf_classifier.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Classification Report for Testing Data:\n", classification_report(Y_test, Y_test_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non GDM', 'GDM'], yticklabels=['Non GDM', 'GDM'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = rf_classifier.best_estimator_.feature_importances_
features = diabetes_dataset.columns.drop(['Class Label(GDM /Non GDM)', 'Case Number'])
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances')
plt.show()

# Function to make predictions for new data with probabilities
def predict_diabetes_rf(input_data):
    standardized_input = scaler.transform([input_data])
    probability = rf_classifier.predict_proba(standardized_input)[0][1]  # Probability of having GDM

    if probability < 0.30:
        risk_level = "Low Risk"
    elif 0.30 <= probability <= 0.60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return f"The model predicts a {risk_level} with a probability of {probability:.2f} of having gestational diabetes."

# Function to read input data from Excel, make predictions, and export results to CSV
def predict_from_excel(input_file, output_file):
    # Read input data from Excel file
    input_data = pd.read_excel(input_file)
    
    # Standardize the input features
    input_features = scaler.transform(input_data)
    
    # Make predictions
    predictions = rf_classifier.predict(input_features)
    probabilities = rf_classifier.predict_proba(input_features)[:, 1]
    
    # Create a DataFrame to store results
    results = pd.DataFrame({
        'Person': [f'Person {i + 1}' for i in range(len(input_data))],
        'Prediction': predictions,
        'Probability': probabilities
    })
    
    # Map numerical predictions to categorical values
    results['Prediction'] = results['Prediction'].map({0: 'Non GDM', 1: 'GDM'})
    
    # Assign risk levels
    results['Risk Level'] = results['Probability'].apply(lambda p: "Low Risk" if p < 0.30 else "Moderate Risk" if 0.30 <= p <= 0.60 else "High Risk")
    
    # Save the results to a CSV file
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Input and output file
input_file = "inputData.xlsx"
output_file = "predictions.csv"
predict_from_excel(input_file, output_file)
