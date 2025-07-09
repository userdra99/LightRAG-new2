import streamlit as st
import requests
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.lightrag_config import LightRAGConfig
from config.vllm_config import VLLMConfig

st.set_page_config(
    page_title="LightRAG Multi-Service Application",
    page_icon="🚀",
    layout="wide"
)

def check_service_health(url):
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    st.title("🚀 LightRAG Multi-Service Application")
    
    # Sidebar - Service Status
    st.sidebar.header("🔧 Service Status")
    
    llm_status = check_service_health(VLLMConfig.get_llm_endpoint())
    embed_status = check_service_health(VLLMConfig.get_embedding_endpoint())
    
    llm_icon = "🟢" if llm_status else "🔴"
    embed_icon = "🟢" if embed_status else "🔴"
    
    st.sidebar.write(f"{llm_icon} **LLM Service** ({'Online' if llm_status else 'Offline'})")
    st.sidebar.write(f"   Port: {VLLMConfig.LLM_PORT}")
    st.sidebar.write(f"   Model: {VLLMConfig.LLM_MODEL}")
    
    st.sidebar.write(f"{embed_icon} **Embedding Service** ({'Online' if embed_status else 'Offline'})")
    st.sidebar.write(f"   Port: {VLLMConfig.EMBED_PORT}")
    st.sidebar.write(f"   Model: {VLLMConfig.EMBED_MODEL}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "🔍 Query", "⚙️ Config"])
    
    with tab1:
        st.header("📄 Document Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'docx', 'xlsx', 'csv']
            )
            
            if uploaded_files:
                if st.button("📤 Process Files"):
                    docs_path = LightRAGConfig.get_docssource_path()
                    docs_path.mkdir(parents=True, exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        file_path = docs_path / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"✅ Saved: {uploaded_file.name}")
        
        with col2:
            st.subheader("DOCSSOURCE Directory")
            docs_path = LightRAGConfig.get_docssource_path()
            
            if st.button("📁 Scan Directory"):
                if docs_path.exists():
                    files = list(docs_path.rglob('*'))
                    doc_files = [f for f in files if f.is_file() and LightRAGConfig.is_supported_file(str(f))]
                    
                    if doc_files:
                        st.success(f"Found {len(doc_files)} documents:")
                        for doc_file in doc_files[:5]:  # Show first 5
                            size = doc_file.stat().st_size
                            st.write(f"• {doc_file.name} ({size:,} bytes)")
                        if len(doc_files) > 5:
                            st.caption(f"... and {len(doc_files) - 5} more")
                    else:
                        st.info("No supported documents found")
                else:
                    st.warning("DOCSSOURCE directory doesn't exist")
            
            st.caption(f"Directory: {docs_path}")
    
    with tab2:
        st.header("🔍 Query Knowledge Base")
        
        if not (llm_status and embed_status):
            st.error("⚠️ Both LLM and Embedding services must be running to process queries")
            return
        
        query_text = st.text_area(
            "Enter your question:",
            placeholder="Ask a question about your documents...",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            query_mode = st.selectbox(
                "Query Mode",
                ["hybrid", "local", "global", "naive"],
                index=0
            )
        
        with col2:
            if st.button("🚀 Run Query", disabled=not query_text.strip()):
                with st.spinner("Processing query..."):
                    st.info("Query functionality will be implemented when LightRAG is fully integrated")
                    st.write(f"**Query:** {query_text}")
                    st.write(f"**Mode:** {query_mode}")
    
    with tab3:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("vLLM Configuration")
            st.code(f"""
LLM Model: {VLLMConfig.LLM_MODEL}
LLM Endpoint: {VLLMConfig.get_llm_endpoint()}
Embedding Model: {VLLMConfig.EMBED_MODEL}
Embedding Endpoint: {VLLMConfig.get_embedding_endpoint()}
            """)
        
        with col2:
            st.subheader("LightRAG Configuration")
            st.code(f"""
Working Directory: {LightRAGConfig.WORKING_DIR}
DOCSSOURCE Path: {LightRAGConfig.DOCSSOURCE_PATH}
Chunk Size: {LightRAGConfig.CHUNK_SIZE}
Embedding Dimension: {LightRAGConfig.EMBEDDING_DIM}
            """)

if __name__ == "__main__":
    main()
