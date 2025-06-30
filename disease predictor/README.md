# 🧬 Disease Prediction System (Streamlit + Scikit-learn)

This is a Machine Learning-based disease prediction web app built using **Scikit-learn** and **Streamlit**.  
Users can select symptoms and the model will predict the most probable disease based on training data.

## 🔍 Features
- ML model trained on symptom-disease dataset
- Streamlit-based frontend with multiselect UI
- Real-time disease prediction
- Easy to deploy on Streamlit Cloud

## 📁 Project Structure

├── app.py # Streamlit app frontend
├── train_model.py # Model training script
├── requirements.txt # Python dependencies
├── data/
│ └── Training.csv # Training dataset
├── model/
│ ├── model.pkl
│ ├── label_encoder.pkl
│ └── vectorizer.pkl