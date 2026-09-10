import streamlit as st

st.set_page_config(page_title="FactoryOS AI Copilot", layout="wide")

st.title("🏭 FactoryOS AI Operations Copilot")

st.markdown("""
Ask questions about factory operations, suppliers, SOPs, production reports, and operational documents.
""")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

question = st.text_input(
    "Ask a question about your factory document"
)

if st.button("Get Answer"):
    if question:
        st.info("RAG engine will answer here in the next version.")
