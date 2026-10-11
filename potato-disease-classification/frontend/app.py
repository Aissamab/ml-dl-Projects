import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.title("🥔 Potato Leaf Disease Classifier")

uploaded = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, width=300)
    if st.button("Predict"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        r = requests.post(API_URL, files=files)
        if r.ok:
            result = r.json()
            st.success(f"{result['class']} ({result['confidence']:.1%})")
        else:
            st.error(f"API error {r.status_code}")