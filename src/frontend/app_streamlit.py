import streamlit as st
import requests

st.title("Prediction interface")

review = st.text_input("Enter your review:")

if st.button("Predict"):
    #url = "http://127.0.0.1:8000/predict/"
    url = "http://backend:8000/predict/" # Use the service name defined in docker-compose.yml
    response = requests.post(url, json={"text": [review]})
    st.write(f"Prediction : {response.json()['prediction']}")