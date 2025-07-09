import asyncio
import sys
from pathlib import Path
import openai
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from config.lightrag_config import LightRAGConfig
from config.vllm_config import VLLMConfig

# Configure OpenAI clients globally
llm_client = openai.OpenAI(
    base_url=VLLMConfig.get_llm_endpoint(),
    api_key="vllm-placeholder"
)

embed_client = openai.OpenAI(
    base_url=VLLMConfig.get_embedding_endpoint(),
    api_key="vllm-placeholder"
)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """LLM function compatible with LightRAG expectations"""
    print(f"🔍 LLM called with prompt length: {len(str(prompt))}, system_prompt: {system_prompt is not None}, history_messages: {type(history_messages)}, kwargs: {list(kwargs.keys())}")

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history_messages:
        messages.extend(history_messages)

    messages.append({"role": "user", "content": prompt})

    response = llm_client.chat.completions.create(
        model="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        messages=messages,
        max_tokens=kwargs.get('max_tokens', 1000),
        temperature=kwargs.get('temperature', 0.1),
        top_p=kwargs.get('top_p', 1.0),
        n=kwargs.get('n', 1)
    )
    return response.choices[0].message.content

async def embedding_func(texts) -> np.ndarray:
    """Embedding function compatible with LightRAG expectations"""
    if isinstance(texts, str):
        texts = [texts]

    response = embed_client.embeddings.create(
        model="jinaai/jina-embeddings-v4-vllm-retrieval",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

async def setup_lightrag():
    """Initialize LightRAG with vLLM endpoints"""
    working_dir = LightRAGConfig.get_working_dir()
    working_dir.mkdir(parents=True, exist_ok=True)

    # Get embedding dimension from a test call
    test_embedding = await embedding_func(["test"])
    embedding_dimension = test_embedding.shape[1]
    print(f"📏 Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    # Initialize storages
    await rag.initialize_storages()

    # Initialize pipeline status with history_messages
    await initialize_pipeline_status()

    return rag

def extract_pdf_content(doc_file):
    """Extract content from PDF file"""
    try:
        import PyPDF2
        with open(doc_file, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            content = "\n".join(page.extract_text() or "" for page in reader.pages if page.extract_text())
        return content
    except Exception as e:
        print(f"Warning: Could not extract PDF content from {doc_file}: {e}")
        return ""

async def process_documents():
    """Process documents from DOCSSOURCE"""

    print("🚀 Initializing LightRAG...")
    rag = await setup_lightrag()

    docs_path = LightRAGConfig.get_docssource_path()

    if not docs_path.exists():
        print("❌ DOCSSOURCE directory not found")
        return None

    # Find all supported documents
    doc_files = []
    for ext in ['.pdf']:   # Start with PDF only
        doc_files.extend(docs_path.glob(f"*{ext}"))

    if not doc_files:
        print("📭 No documents found in DOCSSOURCE")
        return rag

    print(f"📄 Found {len(doc_files)} documents to process")

    for doc_file in doc_files:
        print(f"📖 Processing: {doc_file.name}")

        try:
            # Extract document content
            content = extract_pdf_content(doc_file)

            if not content or content.strip() == "":
                print(f"⚠️  No content extracted from {doc_file.name}")
                continue

            # Insert into LightRAG using async method
            print(f"🔄 Inserting content with {len(content)} characters...")
            await rag.ainsert(content)
            print(f"✅ Processed: {doc_file.name}")

        except Exception as e:
            print(f"❌ Error processing {doc_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print("🎉 Document processing complete!")
    return rag

async def query_documents(rag, query, mode="hybrid"):
    """Query the processed documents"""
    try:
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        return result
    except Exception as e:
        return f"Error processing query: {e}"

if __name__ == "__main__":
    async def main():
        print("🔧 Starting LightRAG Document Processor...")
        print("📡 Connecting to vLLM services...")

        # Test vLLM connections first
        try:
            models = llm_client.models.list()
            print(f"✅ LLM service connected: {len(models.data)} models available")

            embed_models = embed_client.models.list()
            print(f"✅ Embedding service connected: {len(embed_models.data)} models available")

        except Exception as e:
            print(f"❌ Failed to connect to vLLM services: {e}")
            print("Please ensure both vLLM services are running on ports 8000 and 8001")
            return

        rag = await process_documents()

        if rag:
            print("\n🔍 You can now query your documents!")
            print("Example queries:")
            print("- What is this document about?")
            print("- Summarize the main points")
            print("- What are the key findings?")

            # Example query
            query = "What is this document about?"
            print(f"\n🤔 Querying: {query}")
            result = await query_documents(rag, query)
            print(f"📝 Answer: {result}")

            # Keep the service running
            print("\n🔄 LightRAG processor running - ready to serve queries...")
            print("Press Ctrl+C to stop")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 LightRAG processor stopped")

    asyncio.run(main())