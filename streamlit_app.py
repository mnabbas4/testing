import streamlit as st
import requests
import json

# 🔗 Replace with your actual ngrok public URL
BACKEND_URL = "https://83c00fb037c5.ngrok-free.app"

st.title("AI Project Planner (Remote Aspose Server)")

st.markdown("### Enter your project structure text")
user_input = st.text_area(
    "Paste your project outline or JSON structure here:",
    height=250,
    placeholder="Example:\n{\n  'project_name': 'AI Workflow Builder',\n  'tasks': [...]\n}"
)

if st.button("Generate MS Project XML"):
    if not user_input.strip():
        st.warning("⚠️ Please enter your project structure text first.")
    else:
        try:
            # Try to interpret as JSON if possible
            try:
                data = json.loads(user_input)
            except json.JSONDecodeError:
                # if plain text, send as raw text
                data = {"raw_text": user_input}

            st.info("⏳ Sending to Aspose backend... please wait...")

            response = requests.post(f"{BACKEND_URL}/generate_project", json=data)

            if response.status_code == 200:
                xml_data = response.content
                st.success("✅ MS Project XML generated successfully!")
                st.download_button(
                    "Download MS Project XML",
                    xml_data,
                    file_name="project_output.xml",
                    mime="application/xml"
                )
            else:
                st.error(f"❌ Server error: {response.text}")
        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")
