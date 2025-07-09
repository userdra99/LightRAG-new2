#!/bin/bash

# =============================================================================
# LightRAG Application Startup Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check if service is ready
check_service_ready() {
    local url=$1
    local service_name=$2
    local max_attempts=60  # 5 minutes max (5 seconds * 60)
    local attempt=1
    
    echo "Checking if $service_name is ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/v1/models" > /dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within 5 minutes"
    return 1
}

# Function to check if Streamlit is ready
check_streamlit_ready() {
    local max_attempts=12  # 1 minute max (5 seconds * 12)
    local attempt=1
    
    echo "Checking if Streamlit is ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:8501" > /dev/null 2>&1; then
            print_status "Streamlit is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "Streamlit failed to start within 1 minute"
    return 1
}

# Start vLLM services
start_vllm() {
    print_info "Starting vLLM LLM service (port 8000)..."
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Start LLM service
    nohup venv/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16 \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype float16 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.6 \
        > logs/vllm_llm.log 2>&1 &
    
    LLM_PID=$!
    echo "LLM service started with PID: $LLM_PID"
    
    # Wait for LLM service to be ready
    if ! check_service_ready "http://localhost:8000" "LLM service"; then
        print_error "Failed to start LLM service"
        exit 1
    fi
    
    print_info "Starting vLLM Embedding service (port 8001)..."
    
    # Start Embedding service
    nohup venv/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model jinaai/jina-embeddings-v4-vllm-retrieval \
        --port 8001 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.3 \
        --max-model-len 8192 \
        --task embedding \
        > logs/vllm_embedding.log 2>&1 &
    
    EMBED_PID=$!
    echo "Embedding service started with PID: $EMBED_PID"
    
    # Wait for Embedding service to be ready
    if ! check_service_ready "http://localhost:8001" "Embedding service"; then
        print_error "Failed to start Embedding service"
        exit 1
    fi
    
    print_status "Both vLLM services are ready and running!"
    echo "LLM PID: $LLM_PID, Embedding PID: $EMBED_PID"
    
    # Save PIDs for stop script
    echo "$LLM_PID" > logs/llm_pid.txt
    echo "$EMBED_PID" > logs/embedding_pid.txt
}

# Start LightRAG
start_lightrag() {
    print_info "Starting LightRAG document processor..."

    # Activate the virtual environment
    source venv/bin/activate

    # Run the LightRAG processor
    nohup venv/bin/python3 lightrag_processor.py > logs/lightrag.log 2>&1 &
    
    LIGHTRAG_PID=$!
    echo "LightRAG started with PID: $LIGHTRAG_PID"

    # Wait for LightRAG to start processing
    echo "Waiting for LightRAG to initialize..."
    sleep 15  # Give it time to connect to vLLM and start processing

    if ! ps -p "$LIGHTRAG_PID" > /dev/null 2>&1; then
        print_error "LightRAG did not start successfully"
        exit 1
    fi

    print_status "LightRAG processing started."
    echo "$LIGHTRAG_PID" > logs/lightrag_pid.txt
}

# Start Streamlit
start_streamlit() {
    print_info "Starting Streamlit application..."

    # Activate the virtual environment if not already active
    source venv/bin/activate

    # Run the Streamlit app
    nohup venv/bin/streamlit run lightrag_app.py > logs/streamlit.log 2>&1 &
    
    STREAMLIT_PID=$!
    echo "Streamlit started with PID: $STREAMLIT_PID"

    # Wait for Streamlit to be ready
    if ! check_streamlit_ready; then
        print_error "Failed to start Streamlit"
        exit 1
    fi

    print_status "Streamlit application started and ready!"
    echo "$STREAMLIT_PID" > logs/streamlit_pid.txt
}

# Display final status
show_status() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}🎉 Application Started Successfully!${NC}"
    echo "=========================================="
    echo ""
    echo "Services running:"
    echo "🔸 vLLM LLM Service:      http://localhost:8000"
    echo "🔸 vLLM Embedding:       http://localhost:8001"
    echo "🔸 LightRAG Processor:   Background processing"
    echo "🔸 Streamlit Interface:  http://localhost:8501"
    echo ""
    echo "Log files:"
    echo "📄 vLLM LLM:     logs/vllm_llm.log"
    echo "📄 vLLM Embed:   logs/vllm_embedding.log"
    echo "📄 LightRAG:     logs/lightrag.log"
    echo "📄 Streamlit:    logs/streamlit.log"
    echo ""
    echo "To stop all services: ./stop.sh"
    echo ""
}

# Main startup sequence
main() {
    echo -e "${BLUE}🚀 Starting LightRAG Application${NC}"
    echo "Sequential startup: vLLM → LightRAG → Streamlit"
    echo "This may take several minutes for vLLM to load models..."
    echo ""
    
    start_vllm
    echo ""
    start_lightrag  
    echo ""
    start_streamlit
    echo ""
    show_status
}

# Run main function
main "$@"
