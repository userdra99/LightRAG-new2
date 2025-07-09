import openai
import sys

def test_vllm_connections():
    """Test vLLM API connections"""
    
    print("🔧 Testing vLLM API connections...")
    
    # Test LLM service
    try:
        llm_client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="test"
        )
        models = llm_client.models.list()
        print(f"✅ LLM service connected: {len(models.data)} models available")
        print(f"   Available models: {[m.id for m in models.data]}")
        
        # Test a simple completion
        response = llm_client.chat.completions.create(
            model="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
            messages=[{"role": "user", "content": "Hello! Say hi back."}],
            max_tokens=50
        )
        print(f"✅ LLM completion test: {response.choices[0].message.content.strip()}")
        
    except Exception as e:
        print(f"❌ LLM service failed: {e}")
        return False
    
    # Test Embedding service
    try:
        embed_client = openai.OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="test"
        )
        embed_models = embed_client.models.list()
        print(f"✅ Embedding service connected: {len(embed_models.data)} models available")
        print(f"   Available models: {[m.id for m in embed_models.data]}")
        
        # Test a simple embedding
        response = embed_client.embeddings.create(
            model="jinaai/jina-embeddings-v4-vllm-retrieval",
            input=["Hello world"]
        )
        print(f"✅ Embedding test: Got embedding with {len(response.data[0].embedding)} dimensions")
        
    except Exception as e:
        print(f"❌ Embedding service failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_vllm_connections()
    if success:
        print("\n🎉 All vLLM services are working correctly!")
    else:
        print("\n❌ One or more vLLM services have issues.")
        sys.exit(1)
