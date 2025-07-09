#!/bin/bash

# =============================================================================
# LightRAG Application Stop Script
# =============================================================================

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

# Function to kill process by PID file
kill_by_pid_file() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null
            sleep 2
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null
                print_warning "$service_name force-killed (PID: $pid)"
            else
                print_status "$service_name stopped (PID: $pid)"
            fi
        else
            print_info "$service_name was not running (PID: $pid)"
        fi
        rm -f "$pid_file"
    else
        print_info "No PID file found for $service_name"
    fi
}

# Function to kill processes by name pattern
kill_by_pattern() {
    local pattern=$1
    local service_name=$2
    
    local pids=$(pgrep -f "$pattern" || true)
    if [ -n "$pids" ]; then
        echo "$pids" | while read -r pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                kill "$pid" 2>/dev/null
                sleep 1
                
                # Force kill if still running
                if ps -p "$pid" > /dev/null 2>&1; then
                    kill -9 "$pid" 2>/dev/null
                    print_warning "$service_name force-killed (PID: $pid)"
                else
                    print_status "$service_name stopped (PID: $pid)"
                fi
            fi
        done
    else
        print_info "No running $service_name processes found"
    fi
}

# Stop services in reverse order
stop_streamlit() {
    print_info "Stopping Streamlit..."
    kill_by_pid_file "logs/streamlit_pid.txt" "Streamlit"
    kill_by_pattern "streamlit run lightrag_app.py" "Streamlit"
}

stop_lightrag() {
    print_info "Stopping LightRAG..."
    kill_by_pid_file "logs/lightrag_pid.txt" "LightRAG"
    kill_by_pattern "lightrag_processor.py" "LightRAG"
}

stop_vllm() {
    print_info "Stopping vLLM services..."
    
    # Stop LLM service
    kill_by_pid_file "logs/llm_pid.txt" "vLLM LLM"
    
    # Stop Embedding service
    kill_by_pid_file "logs/embedding_pid.txt" "vLLM Embedding"
    
    # Kill any remaining vLLM processes
    kill_by_pattern "vllm.entrypoints.openai.api_server.*8000" "vLLM LLM (port 8000)"
    kill_by_pattern "vllm.entrypoints.openai.api_server.*8001" "vLLM Embedding (port 8001)"
}

# Check if any services are running
check_services() {
    echo ""
    print_info "Checking for running services..."
    
    local running=false
    
    if pgrep -f "streamlit run lightrag_app.py" > /dev/null; then
        echo "🔸 Streamlit is running"
        running=true
    fi
    
    if pgrep -f "lightrag_processor.py" > /dev/null; then
        echo "🔸 LightRAG is running"
        running=true
    fi
    
    if pgrep -f "vllm.entrypoints.openai.api_server.*8000" > /dev/null; then
        echo "🔸 vLLM LLM (port 8000) is running"
        running=true
    fi
    
    if pgrep -f "vllm.entrypoints.openai.api_server.*8001" > /dev/null; then
        echo "🔸 vLLM Embedding (port 8001) is running"
        running=true
    fi
    
    if [ "$running" = false ]; then
        print_info "No services are currently running"
    fi
    
    echo ""
}

# Main stop function
main() {
    echo -e "${BLUE}🛑 Stopping LightRAG Application${NC}"
    echo "Shutdown order: Streamlit → LightRAG → vLLM"
    echo ""
    
    # Check what's running before stopping
    check_services
    
    # Stop services in reverse order
    stop_streamlit
    echo ""
    
    stop_lightrag
    echo ""
    
    stop_vllm
    echo ""
    
    # Final check
    print_info "Verifying all services stopped..."
    sleep 2
    check_services
    
    print_status "All services have been stopped!"
    
    # Clean up log directory
    if [ "$1" = "--clean-logs" ]; then
        print_info "Cleaning log files..."
        rm -f logs/*.log logs/*_pid.txt
        print_status "Log files cleaned"
    fi
}

# Run main function
main "$@"
