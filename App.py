import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title='Water Quality Prediction', page_icon=':droplet:', layout='wide')

# Load models, scaler, and polynomial features transformer
model_dict = {name: joblib.load(f"{name.replace(' ', '_')}.joblib") for name in
              ['Logistic Regression', 'SVM', 'Decision Tree']}
scaler = joblib.load('scaler.joblib')
poly = joblib.load('poly.joblib')  # Load polynomial features transformer


def predict_quality(model,  model_name, input_data):
    input_data_scaled = scaler.transform([input_data])
    input_data_poly = poly.transform(input_data_scaled)  # Apply polynomial transformation
    prediction = model.predict(input_data_poly)
    if model_name != 'SVM':
        probability = model.predict_proba(input_data_poly)[0]
        return prediction[0], max(probability)
    else:
        return prediction[0], None


def show_falling_droplets():
    droplet_html = """
    <style>
    .rain {
        position: relative;
        display: block;
        width: 100%;
        height: 100px;
        overflow: hidden;
        background: transparent;
    }
    .drop {
        background: #29abe2;
        border-radius: 50%;
        position: absolute;
        bottom: 100%;
        width: 10px;
        height: 10px;
        animation: fall .63s linear infinite;
    }
    @keyframes fall {
        to {
            transform: translateY(100px);
        }
    }
    </style>
    <div class="rain">
        <div class="drop" style="left: 10%; animation-delay: 0s;"></div>
        <div class="drop" style="left: 20%; animation-delay: .2s;"></div>
        <div class="drop" style="left: 40%; animation-delay: .4s;"></div>
        <div class="drop" style="left: 60%; animation-delay: .6s;"></div>
        <div class="drop" style="left: 80%; animation-delay: .8s;"></div>
    </div>
    """
    st.markdown(droplet_html, unsafe_allow_html=True)


def show_prediction_page():
    st.write('## Water Quality Prediction')
    st.write('---')

    selected_model = st.selectbox('Select the model to use for prediction:', list(model_dict.keys()))
    model = model_dict[selected_model]

    ph = st.number_input('pH Level:', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    Hardness = st.number_input('Hardness:', min_value=0, max_value=500, value=150, step=1)
    Solids = st.number_input('Solids:', min_value=0, max_value=100000, value=50000, step=100)
    Chloramines = st.number_input('Chloramines:', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    Sulfate = st.number_input('Sulfate:', min_value=0, max_value=1000, value=250, step=10)
    Conductivity = st.number_input('Conductivity:', min_value=0, max_value=1000, value=500, step=1)
    Organic_carbon = st.number_input('Organic Carbon:', min_value=0, max_value=100, value=50, step=1)
    Trihalomethanes = st.number_input('Trihalomethanes:', min_value=0.0, max_value=200.0, value=100.0, step=0.1)
    Turbidity = st.number_input('Turbidity:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    input_data = [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]

    if st.button('Predict Water Quality'):
        prediction, probability = predict_quality(model, selected_model, input_data)
        result = "Potable" if prediction == 1 else "Not Potable"
        st.write(f"**Prediction:** {result}")

        if probability is not None:
            st.write(f"**Prediction Probability:** {probability * 100:.2f}% Potable")
        else:
            st.write("Probability estimates not available for SVM.")

        if prediction == 1:
            show_falling_droplets()


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Welcome", "Predict"])

    if choice == "Welcome":
        st.title("Welcome to the Water Quality Prediction App")
        st.image("Wave-Wallpaper-HD.jpg", use_column_width=True)  # Specify your image path here
        st.write("## Team Members:")
        st.write("Roa Alaa, Amira Ashraf, Malak Mohand, Malak Gholam, Ali Ezzat, Salah-Eldeen Tarek")
    elif choice == "Predict":
        show_prediction_page()


if __name__ == "__main__":
    main()
