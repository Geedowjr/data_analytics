import streamlit as st
import pandas as pd
import joblib

model = joblib.load("car_price_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.title("Bilpris-prediktion")

year = st.number_input("Year", 2000, 2025, 2020)
engine_size = st.number_input("Engine Size", 1.0, 6.0, 2.0)
mileage = st.number_input("Mileage", 0, 300000, 50000)
doors = st.number_input("Doors", 2, 5, 4)
owner_count = st.number_input("Owner Count", 1, 5, 1)

brand = st.selectbox("Brand", ["Audi", "BMW", "Ford", "Honda", "Toyota"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Automatic"])
model_name = st.selectbox("Model", ["A3", "A4", "Focus", "Civic", "Corolla"])

if st.button("Prediktera pris"):
    new_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Year": [year],
        "Engine_Size": [engine_size],
        "Fuel_Type": [fuel_type],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Doors": [doors],
        "Owner_Count": [owner_count]
    })

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(new_data)[0]

    st.success(f"Predikterat pris: {prediction:.0f}")