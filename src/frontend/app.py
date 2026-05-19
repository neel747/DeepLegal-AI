import streamlit as st
import requests
import json
import os

# Page Config
st.set_page_config(
    page_title="DeepLegal AI",
    page_icon="⚖️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #1e2130;
        border-left: 5px solid #4a90e2;
        padding: 10px;
        font-size: 0.85em;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Sidebar
with st.sidebar:
    st.title("⚖️ DeepLegal AI")
    st.markdown("---")
    
    st.subheader("📁 Upload Contract")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Uploading and starting pipeline..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
                if response.status_code == 200:
                    st.success(f"Successfully uploaded {uploaded_file.name}!")
                else:
                    st.error("Upload failed.")

    st.markdown("---")
    st.subheader("📊 System Stats")
    # Mock stats for demo
    st.metric("Context Recall", "98%")
    st.metric("Avg. Response Time", "1.2s")
    st.metric("Graphs Nodes", "1,240")

# Main Chat Interface
st.title("Contract Intelligence & Reasoning")
st.info("Ask complex multi-hop questions about your legal documents. The system uses LangGraph to reason and self-correct.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Citations"):
                for i, src in enumerate(message["sources"]):
                    section_val = src.get('section') or src.get('metadata', {}).get('section') or 'Unknown'
                    st.markdown(f"**Source {i+1}:** {src.get('source')} (Section {section_val})")
                    st.caption(src.get('content')[:300] + "...")

# Chat Input
if prompt := st.chat_input("What is the liability cap in the MSA?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Reasoning..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    full_response = data["answer"]
                    sources = data["sources"]
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Rerender to show expander
                    st.rerun()
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
