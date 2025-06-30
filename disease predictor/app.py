import streamlit as st
import joblib

# Load model files
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# List of all symptoms (you can expand this from your dataset)
symptom_list = [
    'abdominal_pain', 'acidity', 'anxiety', 'back_pain', 'blackheads',
    'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool',
    'blurred_and_distorted_vision', 'breathlessness', 'chest_pain',
    'chills', 'constipation', 'continuous_sneezing', 'cough',
    'cramps', 'dark_urine', 'dehydration', 'diarrhoea', 'dizziness',
    'excessive_hunger', 'fatigue', 'headache', 'high_fever', 'indigestion',
    'irregular_sugar_level', 'itching', 'joint_pain', 'lethargy',
    'loss_of_appetite', 'low_body_temp', 'mood_swings', 'muscle_pain',
    'nausea', 'pain_behind_the_eyes', 'patches_in_throat', 'restlessness',
    'runny_nose', 'shivering', 'skin_rash', 'slurred_speech', 'sore_throat',
    'stomach_pain', 'sunken_eyes', 'sweating', 'vomiting', 'weakness'
    # ✨ Add more symptoms based on your dataset!
]

# Streamlit Page Settings
st.set_page_config(page_title="🧠 Disease Predictor", layout="centered")
st.title("🧬 Disease Prediction System")
st.markdown("🔍 **Select symptoms from the list to predict a possible disease.**")

# Symptom selection UI
selected_symptoms = st.multiselect("🩺 Choose your symptoms", sorted(symptom_list))

# Predict Button
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Join symptoms into a space-separated string
        symptom_input = " ".join(selected_symptoms)
        X_input = vectorizer.transform([symptom_input])
        prediction = model.predict(X_input)
        disease = le.inverse_transform(prediction)[0]

        st.success(f"✅ Based on the symptoms, the predicted disease is: **{disease}**")
