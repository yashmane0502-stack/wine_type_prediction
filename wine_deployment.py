import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("dtc_model.pkl")

st.title("Wine Type Prediction ")

# Inputs
fixed_acidity = st.number_input("fixed_acidity")
volatile_acidity = st.number_input("volatile_acidity")
citric_acid = st.number_input("citric_acid")
residual_sugar = st.number_input("residual_sugar")
chlorides  = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free_sulfur_dioxide")
total_sulfur_dioxide = st.number_input("total_sulfur_dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates  = st.number_input("sulphates")
alcohol = st.number_input("alcohol")

# Create DataFrame (❌ removed "quality")
input_data = pd.DataFrame([{
    "fixed_acidity": fixed_acidity,
    "volatile_acidity": volatile_acidity,
    "citric_acid": citric_acid,
    "residual_sugar": residual_sugar,
    "chlorides": chlorides,
    "free_sulfur_dioxide": free_sulfur_dioxide,
    "total_sulfur_dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol
}])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]

        if prediction == "white":
            st.success(" White Wine")
        else:
            st.error(" Red Wine")

    except Exception as e:
        st.error(f"Error: {e}")
