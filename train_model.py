import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("diabetes.csv")

# 2. Split into features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save the trained model & feature columns
joblib.dump(model, "diabetes_nb_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("\nModel and columns saved successfully!")
