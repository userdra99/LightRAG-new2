#!/usr/bin/env python3
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logging
from config.lightrag_config import LightRAGConfig
from config.vllm_config import VLLMConfig

async def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting LightRAG Multi-Service Application")
    
    # Ensure directories exist
    LightRAGConfig.ensure_directories()
    logger.info("📁 Directories created/verified")
    
    print("\n🎉 LightRAG Application Started!")
    print("📊 Configuration:")
    print(f"   - Working dir: {LightRAGConfig.get_working_dir()}")
    print(f"   - Documents: {LightRAGConfig.get_docssource_path()}")
    print(f"   - LLM model: {VLLMConfig.LLM_MODEL}")
    print(f"   - Embedding model: {VLLMConfig.EMBED_MODEL}")
    print(f"   - LLM endpoint: {VLLMConfig.get_llm_endpoint()}")
    print(f"   - Embedding endpoint: {VLLMConfig.get_embedding_endpoint()}")
    
    print("\nNext steps:")
    print("1. Add documents to DOCSSOURCE/ directory")
    print("2. Start vLLM services manually:")
    print(f"   - LLM: python -m vllm.entrypoints.openai.api_server --model {VLLMConfig.LLM_MODEL} --port {VLLMConfig.LLM_PORT}")
    print(f"   - Embedding: python -m vllm.entrypoints.openai.api_server --model {VLLMConfig.EMBED_MODEL} --port {VLLMConfig.EMBED_PORT}")
    print("3. Process documents with LightRAG")
    print("4. Start Streamlit UI")
    
    # Keep the application running
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
