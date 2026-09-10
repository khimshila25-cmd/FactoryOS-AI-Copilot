import streamlit as st

st.set_page_config(page_title="FactoryOS AI Copilot")

st.title("🏭 FactoryOS AI Operations Copilot")

st.write("AI-powered manufacturing intelligence assistant.")

query = st.text_input("Ask a question about factory operations")

if query:
    st.success(f"You asked: {query}")
    