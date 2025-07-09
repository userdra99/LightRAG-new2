#!/bin/bash

# Start LLM Service
nohup python -m vllm.entrypoints.openai.api_server \
    --model context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.5 \
    &

# Start Embedding Service
nohup python -m vllm.entrypoints.openai.api_server \
    --model jinaai/jina-embeddings-v4-vllm-retrieval \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 8192 \
    --task embedding \
    &

# Wait a bit to ensure services start
sleep 5

printf "Services started on ports 8000 and 8001.\n"

