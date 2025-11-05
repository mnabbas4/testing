import streamlit as st
import requests
import json

# 🔗 Replace with your actual ngrok public URL (keep updated if it changes!)
BACKEND_URL = "https://83c00fb037c5.ngrok-free.app"

st.title("AI Project Planner (Remote Aspose Server)")

st.markdown("### Enter your project structure text")
user_input = st.text_area(
    "Paste your project outline or JSON structure here:",
    height=250,
    placeholder='Example:\n{\n  "project_name": "AI Workflow Builder",\n  "tasks": [...]\n}'
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
                # if plain text, send as raw text (to be parsed by backend)
                data = {"raw_text": user_input}

            st.info("⏳ Sending to Aspose backend... please wait...")

            # ✅ Correct endpoint:
            response = requests.post(f"{BACKEND_URL}/generate", data={"json_data": json.dumps(data)})

            if response.status_code == 200:
                res_json = response.json()
                download_url = f"{BACKEND_URL}{res_json['download_path']}"
                st.success(f"✅ MS Project XML generated successfully: {res_json['filename']}")
                st.markdown(f"[⬇️ Download XML File]({download_url})", unsafe_allow_html=True)
            else:
                st.error(f"❌ Server error: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")
