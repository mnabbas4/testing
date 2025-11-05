import streamlit as st
import requests
import json

# ✅ Replace this with your ngrok public URL
BACKEND_URL = "https://83c00fb037c5.ngrok-free.app"

st.title("AI Project Planner (Remote Aspose Server)")

uploaded_file = st.file_uploader("Upload project structure (JSON)", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)

    st.write("✅ File uploaded successfully.")
    st.write("Now generating MS Project XML via remote Aspose Server...")

    # Send data to your local Aspose backend
    response = requests.post(f"{BACKEND_URL}/generate_project", json=data)

    if response.status_code == 200:
        xml_data = response.content
        st.download_button(
            "Download MS Project XML",
            xml_data,
            file_name="project_output.xml",
            mime="application/xml"
        )
    else:
        st.error(f"❌ Server error: {response.text}")

