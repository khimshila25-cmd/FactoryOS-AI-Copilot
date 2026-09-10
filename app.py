import streamlit as st

st.set_page_config(page_title="FactoryOS AI Copilot")

st.title("🏭 FactoryOS AI Operations Copilot")

question = st.text_input(
    "Ask a manufacturing question:"
)

if question:
    st.success(
        f"Question received: {question}"
    )
