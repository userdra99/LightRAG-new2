class VLLMConfig:
    LLM_MODEL = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"
    LLM_PORT = 8000
    LLM_HOST = "localhost"
    
    EMBED_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"
    EMBED_PORT = 8001
    EMBED_HOST = "localhost"
    
    @classmethod
    def get_llm_endpoint(cls):
        return f"http://{cls.LLM_HOST}:{cls.LLM_PORT}/v1"
    
    @classmethod
    def get_embedding_endpoint(cls):
        return f"http://{cls.EMBED_HOST}:{cls.EMBED_PORT}/v1"