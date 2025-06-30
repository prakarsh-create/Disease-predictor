import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Load your custom dataset
df = pd.read_csv("Data/dataset.csv")  # make sure filename matches your actual file name

# Fill missing values (optional: you can tweak this based on your needs)
df.fillna("none", inplace=True)

# Encode the symptoms (convert text symptoms to numbers)
symptom_columns = df.columns[1:]  # all Symptom_1 to Symptom_17
for col in symptom_columns:
    df[col] = df[col].astype(str)

# Convert symptoms to one-hot encoded vector strings
# Convert all symptoms to one combined string
df["combined_symptoms"] = df[symptom_columns].apply(lambda row: " ".join(row.values), axis=1)

# Create feature set: use TfidfVectorizer on symptoms
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined_symptoms"])

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model trained and saved using your custom dataset.")
