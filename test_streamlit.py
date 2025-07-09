import streamlit as st

st.title("🚀 LightRAG Test App")
st.write("If you can see this, Streamlit is working!")

st.sidebar.header("Services Status")
st.sidebar.write("✅ LLM Service: Running on port 8000")
st.sidebar.write("✅ Embedding Service: Running on port 8001")
st.sidebar.write("✅ Streamlit: Running on port 8501")

if st.button("Test Button"):
    st.success("Button clicked! Streamlit is working properly.")
